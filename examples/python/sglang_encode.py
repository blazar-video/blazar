#!/usr/bin/env python3
"""Encode SGLang's diffusion output with Blazar, straight off the GPU.

SGLang runs its video pipeline in a *spawned* scheduler process, so the frames
never exist in the parent interpreter. There are two seams, both verified against
sglang 0.5.22 running Wan 2.1 T2V 1.3B:

  DiffusersExecutionStage._postprocess_output
      Checkpoints loaded straight from diffusers. The method's first line is
      ``output = output.cpu().float()``; everything after it (NaN repair, range
      correction, clamp) is a torch op that runs on the GPU just as well, so the
      transfer is the only part worth removing. ``blazar.sglang_interop`` replaces
      this method with one that keeps the tensor on the device and hands it to
      ``Encoder.submit_device_batch``. Reversible (``device_output`` context
      manager); refuses to install if the private method moved rather than
      degrading silently to the host path.

  DecodingStage.decode
      The native Wan pipeline does not use the diffusers stage. ``decode`` returns
      the decoded frames still on the device as ``[1, 3, T, H, W]`` float32 on
      cuda. The sidecar in ``bindings/python/sglang_patch/sitecustomize.py`` hooks
      this one from inside the spawned scheduler (``BLAZAR_SWIZZLE=1`` + that dir
      first on ``PYTHONPATH``), because a monkeypatch in the parent never reaches
      the child.

This script exercises the first seam in-process. Three modes:

  --fake     (default) a stand-in stage with the same method shape as sglang's,
             driven with synthetic device-shaped tensors. On a Mac the tensors
             stay on CPU and the encode uses a software backend, so the swizzle
             path (install -> postprocess_on_device -> submit) is exercised end to
             end without sglang, a model, or a GPU. Proves the hook fires and the
             file decodes; it does not prove zero-copy (that needs CUDA).

  --real     the real seam: imports sglang, installs the device-output hook, and
             runs whatever sglang generation you point it at. Needs a CUDA GPU,
             sglang 0.5.22, and model weights. For the native Wan path use the
             sidecar instead (see --help epilog); this mode drives the
             diffusers-execution-stage seam.

  --from-npy a decoded ``(F, H, W, 3)`` float32 clip in ``[0, 1]`` (THWC, RGB),
             fed through the same postprocess + submit path as a reproducible
             stand-in for a real generation.

Examples:
    python examples/python/sglang_encode.py --fake
    python examples/python/sglang_encode.py --from-npy /tmp/clip.npy --backend nvenc_h264
    python examples/python/sglang_encode.py --real --output /tmp/sglang.mp4
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

import blazar

# The mock backend cannot emit a codec config, so it needs a plausible one to open
# a container. The payload is synthetic, so a mock file is structure-only.
FAKE_AV1C = bytes([0x81, 0x04, 0x0C, 0x00])


def repo_calibrations() -> Path:
    # examples/python/sglang_encode.py -> blazar/ -> blazar-video/calibrations
    return Path(__file__).resolve().parents[2].parent / "calibrations" / "default"


def load_interop():
    """Import ``blazar.sglang_interop`` by path.

    The package ``__init__`` imports the compiled binding; the interop module
    itself takes any object with ``submit_device_batch``, so loading it directly
    keeps ``--fake`` runnable even where the binding is not built for torch.
    """
    path = Path(__file__).resolve().parents[1] / "bindings" / "python" / "python" / "blazar" / "sglang_interop.py"
    if not path.is_file():
        # Installed layout: the module is importable as a submodule.
        from blazar import sglang_interop  # type: ignore

        return sglang_interop
    spec = importlib.util.spec_from_file_location("blazar_sglang_interop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeStage:
    """Same shape as sglang's ``DiffusersExecutionStage`` method.

    The real method copies to host first; this stand-in matches it line for line
    so ``postprocess_on_device`` can be checked against the exact behaviour the
    swizzle replaces.
    """

    def _postprocess_output(self, output):
        import torch

        output = output.cpu().float()
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.5, posinf=1.0, neginf=0.0)
        if output.min() < -0.5 or output.max() > 1.5:
            output = (output + 1) / 2
        return output.clamp(0, 1)


def fake_clip(count: int, width: int, height: int):
    """``(1, 3, T, H, W)`` float32 in ``[0, 1]`` — sglang's post-VAE layout.

    A moving gradient so successive frames differ (a static clip hides B-frame and
    scene-cut bugs behind trivial residuals).
    """
    import torch

    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    clip = torch.empty((1, 3, count, height, width), dtype=torch.float32)
    for t in range(count):
        shift = t / max(count, 1)
        clip[0, 0, t] = (xs / max(width - 1, 1) + shift) % 1.0
        clip[0, 1, t] = (ys / max(height - 1, 1) + shift) % 1.0
        clip[0, 2, t] = 0.5
    return clip


def run_swizzled(interop, encoder, stage, clips, *, fps: float, width: int, height: int) -> dict:
    """Install the device-output hook, push clips through the stage, measure.

    The stage's ``_postprocess_output`` is what sglang calls per request. After
    ``install``, calling it hands the post-processed tensor to ``on_frames``.

    On CUDA the tensor goes to ``submit_device_batch`` with no host copy — the
    seam the swizzle exists for. Off CUDA (a Mac dev box) ``submit_device_batch``
    rejects the tensor by design, so this falls back to the host submit so the
    install/postprocess path and the file are still exercised. The fallback is
    labelled, not silent: ``--fake`` on a Mac proves the hook fires and the file
    decodes, not that the encode was zero-copy.
    """
    seen = {"calls": 0, "frames": 0, "device": False, "host": False}

    def on_frames(frames) -> None:
        seen["calls"] += 1
        count = int(frames.shape[2])
        seen["frames"] += count
        if bool(getattr(frames, "is_cuda", False)):
            batched = frames if frames.is_contiguous() else frames.contiguous()
            encoder.submit_device_batch(batched, fps=fps, range="unit01")
            seen["device"] = True
            return
        # Host fallback: (1, 3, T, H, W) -> per-frame THWC numpy.
        seen["host"] = True
        host = frames[0].permute(1, 2, 3, 0).contiguous().cpu().numpy()
        for t in range(count):
            encoder.submit(
                host[t],
                width=width,
                height=height,
                fps=fps,
                layout="thwc",
                range="unit01",
                scene_cut=(seen["frames"] - count + t == 0),
            )

    restore = interop.install(on_frames, encoder_cls=type(stage))
    started = time.perf_counter()
    try:
        for clip in clips:
            stage._postprocess_output(clip)
    finally:
        restore()
    seen["wall_ms"] = (time.perf_counter() - started) * 1000.0
    return seen


def run_real(args, interop, encoder) -> dict:
    """Install the hook against the real sglang class and run a generation.

    This drives the diffusers-execution-stage seam. For the native Wan pipeline,
    the scheduler runs in a spawned child and the sidecar is the right tool; see
    the epilog. Here we only need the hook installed before sglang runs so each
    request's output is encoded on the device.
    """
    import torch  # noqa: F401  (import early so a missing torch fails clearly)

    # The context manager resolves and patches the real class, and refuses if the
    # private method moved (verified against sglang 0.5.22).
    with interop.device_output(encoder, fps=args.fps, range="unit01"):
        # Placeholder for the caller's sglang generation call. The point of this
        # example is the seam, not a specific prompt: run your normal sglang
        # offline/engine generation here. Each output flows through
        # _postprocess_output -> Blazar on the GPU.
        raise SystemExit(
            "add your sglang generation call inside the device_output block "
            "(e.g. engine.generate(...) for a diffusers-loaded checkpoint). "
            "For the native Wan pipeline use the sitecustomize sidecar instead; "
            "see --help."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=(
            "Native Wan pipeline (spawned scheduler): do NOT use --real. Put "
            "bindings/python/sglang_patch first on PYTHONPATH and set "
            "BLAZAR_SWIZZLE=1, then run sglang normally; sitecustomize.py hooks "
            "DecodingStage.decode inside the child. See that directory's README."
        ),
    )
    parser.add_argument("--fake", action="store_true", help="stand-in stage, synthetic clips (default)")
    parser.add_argument("--from-npy", type=Path, help="(F, H, W, 3) float32 in [0, 1]")
    parser.add_argument("--real", action="store_true", help="patch real sglang and run a generation")
    parser.add_argument("--clips", type=int, default=3, help="fake clips (requests) to push (default 3)")
    parser.add_argument("--frames", type=int, default=9, help="frames per fake clip (default 9)")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/sglang_blazar.mp4"))
    parser.add_argument("--backend", default="svtav1", choices=["svtav1", "nvenc_av1", "nvenc_h264", "rav1e"])
    parser.add_argument("--codec", default=None, help="override codec (defaults by backend)")
    parser.add_argument("--target", default="balanced", choices=["low_latency", "balanced", "quality"])
    parser.add_argument("--calibrations", type=Path, default=repo_calibrations())
    args = parser.parse_args()

    chosen = [m for m in ("fake", "from_npy", "real") if getattr(args, m)]
    if len(chosen) > 1:
        parser.error("choose one of --fake, --from-npy, --real")

    interop = load_interop()

    try:
        import torch
    except ImportError:
        raise SystemExit("this example needs torch (pip install torch)")

    # Codec/extradata: only the mock (rav1e) needs a canned av1C. Real backends
    # derive it from the first keyframe; overriding would write a lie into the
    # container.
    codec = args.codec or {
        "svtav1": "av1",
        "nvenc_av1": "av1",
        "nvenc_h264": "h264",
        "rav1e": "av1",
    }[args.backend]
    extradata = FAKE_AV1C if args.backend == "rav1e" else None
    if extradata is not None:
        print("note: mock backend - output is NOT decodable, structure only")

    calibrations = blazar.Calibrations.load(str(args.calibrations))
    encoder = blazar.Encoder(
        calibrations=calibrations,
        identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
        target=args.target,
        backend=args.backend,
        codec=codec,
        output=str(args.output),
        video_extradata=extradata,
        request_id="sglang-example",
    )

    started = time.perf_counter()
    with encoder as stream:
        if args.real:
            run_real(args, interop, stream)
            frames_total = 0
        else:
            if args.from_npy is not None:
                arr = np.load(args.from_npy)
                if arr.dtype != np.float32 or arr.ndim != 4 or arr.shape[-1] != 3:
                    raise SystemExit(f"{args.from_npy}: expected (F, H, W, 3) float32")
                # THWC [0,1] -> (1, 3, T, H, W) to match the post-VAE layout.
                t = torch.from_numpy(np.ascontiguousarray(arr)).permute(3, 0, 1, 2)
                clips = [t.unsqueeze(0).contiguous()]
                args.height, args.width = arr.shape[1], arr.shape[2]
            else:
                clips = [
                    fake_clip(args.frames, args.width, args.height)
                    for _ in range(args.clips)
                ]

            seen = run_swizzled(interop, stream, FakeStage(), clips,
                                fps=args.fps, width=args.width, height=args.height)
            frames_total = seen["frames"]
            path = "device (zero-copy)" if seen["device"] else "host fallback (no CUDA)"
            print(f"swizzle   : {seen['calls']} stage calls, {seen['frames']} frames "
                  f"via {path}, {seen['wall_ms']:.1f} ms")

        stream.flush(timeout=30.0)
        stats = stream.stats
    elapsed = time.perf_counter() - started

    size = args.output.stat().st_size
    print(f"frames    : {frames_total} ({args.width}x{args.height} @ {args.fps:g} fps)")
    print(f"encoder   : {stats['encoded']} frames, {stats['bytes']} bytes, "
          f"max {stats['max_encode_us'] / 1000:.2f} ms/frame")
    print(f"total     : {elapsed * 1000:.1f} ms")
    print(f"output    : {args.output} ({size} bytes)")
    print("verify    : python examples/python/verify_encode.py "
          f"{args.output} --log <run.log>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
