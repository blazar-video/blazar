#!/usr/bin/env python3
"""Generate with Wan2.1, stream-decode the latents, encode with Blazar.

This is the real-latent version of the streaming test: the latents come from an
actual diffusion rollout, and they are decoded chunk by chunk through the same
`WanStreamingDecoder` that vLLM-Omni's CausVid pipeline uses (it is imported from
a vLLM-Omni checkout and carries session state across chunks).

Why not vLLM-Omni's own AR pipeline: its documented streaming model is
LingBot-World 14B (~28 GB in bf16) and does not fit a 16 GB card. Wan2.1-1.3B
does, and diffusers can hand back the pre-decode latents with
`output_type="latent"`.

`--save-latents` / `--from-latents` cache the rollout so the decode+encode half
can be iterated on in seconds without re-running the model.

Encoding is buffered by default (decode everything, then encode). `--interleave`
encodes while the VAE is still decoding; it used to corrupt the NVENC bitstream
and is now covered by the encoder's one-unit-per-call drain fix.

The scaling step is the one that has to be right: with `output_type="latent"`,
diffusers returns the raw denoised latents, and the pipeline applies
`latents * latents_std + latents_mean` before `vae.decode`. The streaming
decoder expects the already-rescaled values, so this script applies the same
transform with the VAE config's own numbers.

    python examples/python/wan_streaming_generate.py \
        --prompt "a slow dolly through a neon-lit alley in the rain" \
        --height 320 --width 480 --num-frames 21 --num-inference-steps 20 \
        --output /tmp/wan-stream.mp4
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

import blazar


def repo_calibrations() -> Path:
    return Path(__file__).resolve().parents[2].parent / "calibrations" / "default"


def load_streaming_decoder(vllm_omni: Path):
    path = vllm_omni / "vllm_omni" / "experimental" / "ar_diffusion" / "streaming_decode.py"
    if not path.is_file():
        raise SystemExit(f"no streaming decoder at {path}")
    spec = importlib.util.spec_from_file_location("_blazar_streaming_decode", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.WanStreamingDecoder


def chunks(total: int, size: int) -> list[tuple[int, int]]:
    """First chunk is one latent frame, then `size` at a time (AR-style)."""
    spans = [(0, 1)]
    cursor = 1
    while cursor < total:
        step = min(size, total - cursor)
        spans.append((cursor, cursor + step))
        cursor += step
    return spans


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--num-frames", type=int, default=21)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-chunk", type=int, default=2)
    parser.add_argument("--vllm-omni", type=Path, default=Path.home() / "dev/code/vllm-omni")
    parser.add_argument("--output", type=Path, default=Path("/tmp/wan_stream.mp4"))
    parser.add_argument("--backend", default="nvenc_av1", choices=["svtav1", "nvenc_av1", "rav1e"])
    parser.add_argument("--target", default="balanced", choices=["low_latency", "balanced", "quality"])
    parser.add_argument("--calibrations", type=Path, default=repo_calibrations())
    parser.add_argument("--interleave", action="store_true",
                        help="encode while the VAE is still decoding")
    parser.add_argument("--save-latents", type=Path, help="write raw latents here for fast reruns")
    parser.add_argument("--from-latents", type=Path, help="skip generation, load raw latents")
    args = parser.parse_args()

    if not args.from_latents and not args.prompt:
        parser.error("--prompt is required unless --from-latents is given")

    import torch
    from diffusers import AutoencoderKLWan, WanPipeline

    decoder_cls = load_streaming_decoder(args.vllm_omni)

    # Phase 1: real rollout, latents only. CPU offload keeps T5+DiT inside 16 GB.
    if args.from_latents is not None:
        latents = torch.from_numpy(np.load(args.from_latents))
        generation_s = 0.0
        peak = 0.0
        print(f"generated : loaded latents {tuple(latents.shape)} from {args.from_latents}")
    else:
        started = time.perf_counter()
        pipe = WanPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        output = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            output_type="latent",
            generator=torch.Generator(device="cpu").manual_seed(args.seed),
        )
        latents = output.frames
        generation_s = time.perf_counter() - started
        peak = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"generated : latents {tuple(latents.shape)} in {generation_s:.1f} s "
              f"(peak {peak:.2f} GiB reserved)")
        if args.save_latents is not None:
            np.save(args.save_latents, latents.float().cpu().numpy())
            print(f"saved     : latents -> {args.save_latents}")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    # Phase 2: the pipeline's own pre-decode scaling, then streaming decode.
    vae = (
        AutoencoderKLWan.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.bfloat16)
        .to("cuda")
        .eval()
    )
    z_dim = vae.config.z_dim
    device = latents.device
    mean = torch.tensor(
        vae.config.latents_mean, dtype=torch.float32, device=device
    ).view(1, z_dim, 1, 1, 1)
    std = torch.tensor(
        vae.config.latents_std, dtype=torch.float32, device=device
    ).view(1, z_dim, 1, 1, 1)
    latents = (latents.float() * std + mean).to(dtype=torch.bfloat16, device="cuda")

    decoder = decoder_cls(vae)
    state = decoder.new_decode_state("blazar-stream")
    total_latent = latents.shape[2]
    spans = chunks(total_latent, args.latent_chunk)
    print(f"chunks    : {len(spans)} over {total_latent} latent frames {spans}")

    submitted = 0
    decode_s = 0.0
    first_frame_s = None
    encode_started = time.perf_counter()

    calibrations = blazar.Calibrations.load(str(args.calibrations))
    buffered: list[np.ndarray] = []

    def submit(stream, frame):
        nonlocal submitted, first_frame_s
        if not args.interleave:
            buffered.append(frame)
            return
        stream.submit(
            frame,
            width=args.width,
            height=args.height,
            fps=args.fps,
            layout="bcthw",
            range="signed_unit",
            scene_cut=(submitted == 0),
        )
        if first_frame_s is None:
            first_frame_s = time.perf_counter() - encode_started
        submitted += 1

    with blazar.Encoder(
        calibrations=calibrations,
        identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
        target=args.target,
        backend=args.backend,
        output=str(args.output),
        video_extradata=None,
        request_id="wan-streaming",
    ) as stream:
        for index, (lo, hi) in enumerate(spans):
            chunk = latents[:, :, lo:hi]
            began = time.perf_counter()
            with torch.inference_mode():
                decoded = decoder.decode_chunk(chunk, state)
            torch.cuda.synchronize()
            decode_s += time.perf_counter() - began

            host = decoded[0].float().cpu().numpy()  # (3, T, H, W) in [-1, 1]
            frames = host.shape[1]
            for t in range(frames):
                submit(stream, np.ascontiguousarray(host[:, t, :, :]))
            print(f"  chunk {index} latents {lo}:{hi} -> {frames} frames "
                  f"({decode_s * 1000 / (index + 1):.0f} ms/chunk avg)")

        if not args.interleave:
            decoder.release(state)
            torch.cuda.empty_cache()
            for frame in buffered:
                stream.submit(
                    frame,
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                    layout="bcthw",
                    range="signed_unit",
                    scene_cut=(submitted == 0),
                )
                if first_frame_s is None:
                    first_frame_s = time.perf_counter() - encode_started
                submitted += 1
        stream.flush(timeout=120.0)
        stats = stream.stats
    encode_s = time.perf_counter() - encode_started

    decoder.release(state)
    expected = 1 + (total_latent - 1) * vae.config.scale_factor_temporal
    print(f"frames    : {submitted} (expected {expected})")
    if first_frame_s is not None:
        print(f"first frame encoded {first_frame_s * 1000:.1f} ms into the decode phase")
    print(f"decode    : {decode_s * 1000:.0f} ms total")
    print(f"encode    : {stats['encoded']} frames, {stats['bytes']} bytes in "
          f"{encode_s * 1000:.0f} ms (max {stats['max_encode_us'] / 1000:.2f} ms/frame)")
    print(f"output    : {args.output} ({args.output.stat().st_size} bytes)")
    return 0 if submitted == expected else 1


if __name__ == "__main__":
    sys.exit(main())
