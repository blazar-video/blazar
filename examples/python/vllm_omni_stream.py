#!/usr/bin/env python3
"""Stream vLLM-Omni decode chunks into Blazar.

vLLM-Omni's streaming decoder emits ``(1, 3, T, H, W)`` device tensors in
``[-1, 1]`` per autoregressive block. This example drives that decoder and feeds
each decoded chunk to Blazar as it arrives, so time-to-first-encoded-frame
tracks the decode, not the clip length.

Three modes:

  --fake             (default) synthetic chunks on a timer. Same dtype/shape/
                     range as the real seam; runs anywhere, including a Mac.
  --from-npy PATH    a decoded ``(F, H, W, 3)`` float32 array in ``[-1, 1]``.
  --real             the real streaming decoder from a vLLM-Omni checkout
                     (``WanStreamingDecoder``) with real Wan VAE weights on the
                     GPU. Latents come from noise: the AR diffusion transformer
                     is not part of this path, so the pixels are not a model
                     sample. Everything else is the production code: same
                     decoder class, same session state carried across chunks,
                     same device tensor, same host boundary.

The boundary adaptation is one copy::

    chunk = job.decode_chunk(latent, state)      # (1,3,T,H,W) bf16, cuda
    host  = chunk[0].float().cpu().numpy()       # (3,T,H,W) float32

then one Blazar submit per frame with ``layout="bcthw"`` (planar RGB) and
``range="signed_unit"``. In the engine the producer side reaches the same tensor
in ``OmniRequestState.add_multimodal_tensor`` (output_processor.py) or the
``post_decode`` callback of ``experimental/ar_diffusion/streaming_decode.py``,
before ``MultimodalPayload.from_raw`` does its unconditional ``_to_cpu``. Hook
there, not in ``from_raw``.

Real mode needs: `pip install torch diffusers`, a vLLM-Omni checkout, and the Wan
VAE weights (cached from ``Wan-AI/Wan2.1-T2V-1.3B-Diffusers``).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

import blazar

FAKE_AV1C = bytes([0x81, 0x04, 0x0C, 0x00])


def repo_calibrations() -> Path:
    return Path(__file__).resolve().parents[2].parent / "calibrations" / "default"


def fake_chunk(index: int, frames: int, width: int, height: int) -> np.ndarray:
    """(1, 3, T, H, W) float32 in [-1, 1], the streaming decoder's layout."""
    chunk = np.empty((1, 3, frames, height, width), dtype=np.float32)
    for t in range(frames):
        level = -1.0 + 2.0 * ((index * frames + t) % 24) / 23.0
        chunk[0, :, t, :, :] = level
    return chunk


def load_streaming_decoder(vllm_omni: Path):
    """Import ``streaming_decode.py`` directly.

    The module imports only torch (it is written to be exercised without the
    engine), so loading it by path avoids dragging in vllm and its CUDA
    initialization just to get one class.
    """
    path = vllm_omni / "vllm_omni" / "experimental" / "ar_diffusion" / "streaming_decode.py"
    if not path.is_file():
        raise SystemExit(f"no streaming decoder at {path}")
    spec = importlib.util.spec_from_file_location("_blazar_streaming_decode", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules[cls.__module__], so the
    # module has to be registered before its body executes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.WanStreamingDecoder


def run_real(args, stream) -> dict:
    import torch
    from diffusers import AutoencoderKLWan

    decoder_cls = load_streaming_decoder(args.vllm_omni)
    device = args.device
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    vae = (
        AutoencoderKLWan.from_pretrained(
            args.vae_repo, subfolder="vae", torch_dtype=dtype
        )
        .to(device)
        .eval()
    )
    decoder = decoder_cls(vae)
    state = decoder.new_decode_state("blazar-example")

    latent_h = args.height // vae.config.scale_factor_spatial
    latent_w = args.width // vae.config.scale_factor_spatial
    # Session state is resident; admission accounting is per session. The
    # helper needs a measured bytes-per-pixel constant for the checkpoint, so
    # report it only when the caller supplied one.
    try:
        state_bytes = decoder.declared_state_bytes(
            height=args.height, width=args.width, dtype=dtype
        )
    except RuntimeError:
        state_bytes = None

    generator = torch.Generator(device="host").manual_seed(args.seed)
    stats = {"state_bytes": state_bytes, "chunks": 0, "decode_s": 0.0}
    started = time.perf_counter()
    first_submit_s = None
    submitted = 0

    for chunk_index in range(args.chunks):
        latent = torch.randn(
            (1, vae.config.z_dim, args.latent_frames, latent_h, latent_w),
            generator=generator,
            dtype=torch.float32,
        ).to(device=device, dtype=dtype)

        started = time.perf_counter()
        with torch.inference_mode():
            decoded = decoder.decode_chunk(latent, state)
        torch.cuda.synchronize()
        stats["decode_s"] += time.perf_counter() - started
        stats["chunks"] += 1

        # Documented boundary: device -> host, float32, planar RGB.
        host = decoded[0].float().cpu().numpy()
        frames = host.shape[1]
        for t in range(frames):
            frame = np.ascontiguousarray(host[:, t, :, :])
            stream.submit(
                frame,
                width=args.width,
                height=args.height,
                fps=args.fps,
                layout="bcthw",
                range="signed_unit",
                scene_cut=(submitted == 0),
            )
            if first_submit_s is None:
                first_submit_s = time.perf_counter() - started
            submitted += 1

    stats["frames"] = submitted
    stats["first_submit_s"] = first_submit_s
    decoder.release(state)
    stats["decoder_shape"] = tuple(decoded.shape)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fake", action="store_true", help="synthetic chunks (default)")
    parser.add_argument("--from-npy", type=Path, help="decoded (F, H, W, 3) in [-1, 1]")
    parser.add_argument("--real", action="store_true", help="real vLLM-Omni streaming decoder")
    parser.add_argument("--vllm-omni", type=Path, default=Path.home() / "dev/code/vllm-omni")
    parser.add_argument("--vae-repo", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "f32"])
    parser.add_argument("--latent-frames", type=int, default=2, help="latent frames per chunk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--frames-per-chunk", type=int, default=3)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--chunk-delay-ms", type=float, default=40.0,
                        help="fake decode latency between chunks (default 40 ms)")
    parser.add_argument("--output", type=Path, default=Path("/tmp/vllm_omni_blazar.mp4"))
    parser.add_argument("--backend", default="svtav1", choices=["svtav1", "nvenc_av1"])
    parser.add_argument("--target", default="low_latency", choices=["low_latency", "balanced", "quality"])
    parser.add_argument("--calibrations", type=Path, default=repo_calibrations())
    args = parser.parse_args()
    if args.fake and (args.from_npy or args.real):
        parser.error("choose one of --fake, --from-npy, --real")
    if args.from_npy and args.real:
        parser.error("choose one of --from-npy, --real")

    frames: np.ndarray | None = None
    if args.from_npy:
        frames = np.load(args.from_npy)
        if frames.dtype != np.float32 or frames.ndim != 4 or frames.shape[-1] != 3:
            raise SystemExit(f"{args.from_npy}: expected (F, H, W, 3) float32")
        args.height, args.width = frames.shape[1], frames.shape[2]

    # 480x320 is the Wan architecture's native aspect for 1.3B; keep the latent
    # geometry divisible by the VAE's 8x spatial factor.
    if args.real and (args.width % 8 or args.height % 8):
        parser.error("--real needs width and height divisible by 8")

    extradata = None if args.backend == "nvenc" else FAKE_AV1C
    if extradata is not None:
        print("note: mock backend - output is NOT decodable, structure only")

    calibrations = blazar.Calibrations.load(str(args.calibrations))
    submitted = 0
    first_frame_at: float | None = None
    chunk_times: list[float] = []
    started = time.perf_counter()

    with blazar.Encoder(
        calibrations=calibrations,
        identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
        target=args.target,
        backend=args.backend,
        output=str(args.output),
        video_extradata=extradata,
        request_id="vllm-omni-example",
    ) as stream:
        if args.real:
            real = run_real(args, stream)
            submitted = real["frames"]
            print(f"decoder   : WanStreamingDecoder, {real['chunks']} chunks, "
                  f"last output {real['decoder_shape']}")
            print(f"decode    : {real['decode_s'] * 1000:.1f} ms total, "
                  f"{real['decode_s'] / max(real['chunks'], 1) * 1000:.1f} ms/chunk")
            if real["first_submit_s"] is not None:
                print(f"first frame submitted {real['first_submit_s'] * 1000:.1f} ms "
                      f"after the first latent")
            if real["state_bytes"] is not None:
                print(f"session   : {real['state_bytes'] / (1024 * 1024):.1f} MiB "
                      f"resident decoder state")
        else:
            for chunk_index in range(args.chunks):
                chunk_started = time.perf_counter()
                if frames is not None:
                    lo = chunk_index * args.frames_per_chunk
                    hi = min(lo + args.frames_per_chunk, len(frames))
                    if lo >= hi:
                        break
                    chunk = frames[lo:hi]
                    per_frame = [chunk[t] for t in range(len(chunk))]
                else:
                    time.sleep(args.chunk_delay_ms / 1000.0)
                    chunk = fake_chunk(chunk_index, args.frames_per_chunk, args.width, args.height)[0]
                    per_frame = [
                        np.ascontiguousarray(chunk[:, t, :, :].transpose(1, 2, 0))
                        for t in range(chunk.shape[1])
                    ]

                for frame in per_frame:
                    stream.submit(
                        frame,
                        width=args.width,
                        height=args.height,
                        fps=args.fps,
                        layout="thwc",
                        range="signed_unit",
                        scene_cut=(submitted == 0),
                    )
                    submitted += 1
                    if first_frame_at is None:
                        first_frame_at = time.perf_counter() - started
                chunk_times.append((time.perf_counter() - chunk_started) * 1000.0)

        # One barrier before close so the counters reflect everything.
        stream.flush(timeout=30.0)
        stats = stream.stats

    elapsed = time.perf_counter() - started
    print(f"frames    : {submitted} ({args.width}x{args.height} @ {args.fps:g} fps)")
    if args.real is False and first_frame_at is not None:
        print(f"first frame encoded {first_frame_at * 1000:.1f} ms after start")
    if chunk_times:
        print(f"per chunk : {min(chunk_times):.2f}..{max(chunk_times):.2f} ms")
    print(f"encoder   : {stats['encoded']} frames, {stats['bytes']} bytes, "
          f"max {stats['max_encode_us'] / 1000:.2f} ms/frame")
    print(f"total     : {elapsed * 1000:.1f} ms")
    print(f"output    : {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
