#!/usr/bin/env python3
"""Generate with vLLM-Omni and encode the frames with Blazar, in one process.

No intermediate video file: the frames that leave the pipeline go straight into
Blazar's encoder. (The earlier round trip wrote an h264 mp4 with vLLM-Omni, then
decoded it and re-encoded with Blazar, which measured the harness with an extra
lossy generation in the middle.)

Requires the vLLM-Omni stack (`vllm`, `torch`, `diffusers`, the vllm-omni
package) and the Blazar Python binding, both importable from the same
interpreter. On mosk that is `~/.vllm-stack`.

    python examples/python/vllm_omni_generate.py \
        --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
        --prompt "a slow dolly through a neon-lit alley in the rain" \
        --height 320 --width 480 --num-frames 21 --num-inference-steps 20 \
        --fps 16 --output /tmp/wan21-blazar.mp4

The engine is shut down (VRAM freed) before encoding starts. That keeps this
within a 16 GiB card, where the engine alone peaks at ~16.2 GiB. It also means
this run does not overlap generation and encoding; the streaming path is the
separate `--real` mode of vllm_omni_stream.py.

Frames are the pipeline's host output, normalised to [0,1] using the model's
declared tensor range, then submitted as float32 THWC by Blazar. The device
tensor never reaches Blazar here because vLLM-Omni copies to host inside the
pipeline; the zero-copy seam is a separate, later change.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import blazar


def repo_calibrations() -> Path:
    return Path(__file__).resolve().parents[2].parent / "calibrations" / "default"


def frame_array(frames, source_range: str) -> np.ndarray:
    """Normalise the pipeline's output to (F, H, W, 3) float32 in [0, 1]."""
    import torch

    if isinstance(frames, list) and len(frames) == 1:
        frames = frames[0]
    if isinstance(frames, tuple) and len(frames) == 2:
        frames = frames[0]

    def one(item):
        if hasattr(item, "detach"):
            tensor = item.detach().float().cpu()
            if tensor.dim() == 4 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.dim() == 4 and tensor.shape[0] in (3, 4):
                tensor = tensor.permute(1, 2, 3, 0)
            if tensor.dim() == 3 and tensor.shape[0] in (3, 4):
                tensor = tensor.permute(1, 2, 0).unsqueeze(0)
            if source_range == "negative_one_to_one":
                tensor = tensor.clamp(-1, 1) * 0.5 + 0.5
            elif source_range == "zero_to_one":
                tensor = tensor.clamp(0, 1)
            return tensor.numpy()

        array = np.asarray(item)
        if array.ndim == 5:
            array = array[0]
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 4 and array.shape[0] in (3, 4):
            array = array.transpose(1, 2, 3, 0)
        if array.dtype == np.uint8:
            array = array.astype(np.float32) / 255.0
        elif array.dtype != np.float32:
            array = array.astype(np.float32)
        # The engine normalises frames to [0, 1] before returning them (its own
        # export path only applies the model's tensor range to torch.Tensor
        # items), so numpy output must NOT be mapped again: doing that lifts
        # blacks to 0.5 and halves the range. Fail loudly if a pipeline ever
        # hands over raw signed values instead of guessing.
        low = float(array.min())
        if low < -0.01:
            raise SystemExit(
                f"numpy frames have min {low:.3f}, outside [0,1]; the engine is "
                "returning unnormalised tensor-range values, revisit frame_array()"
            )
        np.clip(array, 0.0, 1.0, out=array)
        return array

    if isinstance(frames, (list, tuple)):
        array = np.stack([one(item) for item in frames])
    else:
        array = one(frames)
    if array.ndim == 3:
        array = array[None, ...]
    return np.ascontiguousarray(array, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--num-frames", type=int, default=21)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("/tmp/vllm_omni_blazar.mp4"))
    parser.add_argument("--backend", default="nvenc", choices=["svtav1", "nvenc"])
    parser.add_argument("--target", default="balanced", choices=["low_latency", "balanced", "quality"])
    parser.add_argument("--calibrations", type=Path, default=repo_calibrations())
    parser.add_argument("--keep-engine", action="store_true",
                        help="do not shut the engine down before encoding (needs spare VRAM)")
    args = parser.parse_args()

    # Imported here so --help works without a CUDA stack.
    import torch
    from vllm_omni.diffusion.data import resolve_model_class_name
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_extras import get_model_class_name, get_output_tensor_range

    model_class_name = resolve_model_class_name(args.model)
    source_range = get_output_tensor_range(model_class_name)
    print(f"model     : {args.model} ({model_class_name}, tensor range {source_range})")

    omni = Omni(model=args.model)
    model_class_name = get_model_class_name(omni) or model_class_name
    source_range = get_output_tensor_range(model_class_name)

    prompt = {"prompt": args.prompt, "modalities": ["video"]}
    if args.negative_prompt:
        prompt["negative_prompt"] = args.negative_prompt

    sampling = dict(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
    )
    try:
        params = OmniDiffusionSamplingParams(**sampling, fps=args.fps, frame_rate=args.fps)
    except TypeError:
        params = OmniDiffusionSamplingParams(**sampling)

    started = time.perf_counter()
    output = omni.generate(prompt, params)
    generation_s = time.perf_counter() - started

    if isinstance(output, list):
        output = output[0] if output else None
    if output is None:
        raise SystemExit("vLLM-Omni returned no output")

    frames = output.images
    if isinstance(frames, list) and len(frames) == 1:
        item = frames[0]
        if isinstance(item, tuple):
            frames = item[0]
        elif isinstance(item, dict):
            frames = item.get("frames") or item.get("video")
        else:
            frames = item
    video = frame_array(frames, source_range)
    count, height, width, channels = video.shape
    if channels != 3:
        raise SystemExit(f"expected RGB frames, got {video.shape}")

    peak = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    print(f"generated : {count} frames {width}x{height} in {generation_s:.1f} s "
          f"(peak {peak:.2f} GiB reserved)")

    if not args.keep_engine:
        # Free VRAM before NVENC: the engine alone peaks at the card's limit.
        for method in ("shutdown", "close"):
            close = getattr(omni, method, None)
            if callable(close):
                try:
                    close()
                except Exception as error:  # noqa: BLE001 - best effort teardown
                    print(f"note: {method}() failed: {error}")
                break
        del omni
        torch.cuda.empty_cache()

    calibrations = blazar.Calibrations.load(str(args.calibrations))
    extradata = None if args.backend == "nvenc" else bytes([0x81, 0x04, 0x0C, 0x00])

    encode_started = time.perf_counter()
    with blazar.Encoder(
        calibrations=calibrations,
        identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
        target=args.target,
        backend=args.backend,
        output=str(args.output),
        video_extradata=extradata,
        request_id="vllm-omni-generate",
    ) as stream:
        for index in range(count):
            stream.submit(
                video[index],
                width=width,
                height=height,
                fps=args.fps,
                layout="thwc",
                range="unit01",
                scene_cut=(index == 0),
            )
        stream.flush(timeout=60.0)
        stats = stream.stats
    encode_s = time.perf_counter() - encode_started

    print(f"encoded   : {stats['encoded']} frames, {stats['bytes']} bytes in "
          f"{encode_s * 1000:.0f} ms (max {stats['max_encode_us'] / 1000:.2f} ms/frame)")
    print(f"output    : {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
