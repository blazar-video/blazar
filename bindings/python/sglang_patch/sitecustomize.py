"""Install Blazar's device-output hook inside every process SGLang spawns.

SGLang runs its scheduler in a spawned child, so a patch installed in the parent
never reaches the process holding the frames. Python imports this module at
startup in every interpreter, which is the seam that actually works.

The Wan pipeline does not use `DiffusersExecutionStage` (that path is for models
loaded straight from diffusers); it composes `DecodingStage`, whose `forward`
returns an `OutputBatch` holding the decoded frames while they are still on the
device. That is the hook.

Activated only when BLAZAR_SWIZZLE=1.
"""

import json
import os

if os.environ.get("BLAZAR_SWIZZLE") == "1":
    import threading

    import atexit

    _lock = threading.Lock()
    _state = {"encoder": None, "patched": False}
    _obs = "/var/tmp/blazar/swizzle_obs.jsonl"

    def _patch():
        from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
            DecodingStage,
        )

        original = DecodingStage.decode

        def decode(self, latents, server_args, *args, **kwargs):
            frames = original(self, latents, server_args, *args, **kwargs)
            if frames is None or not hasattr(frames, "is_cuda"):
                return frames
            with _lock:
                record = {
                    "pid": os.getpid(),
                    "shape": list(frames.shape),
                    "dtype": str(frames.dtype),
                    "device": str(frames.device),
                    "contiguous": bool(frames.is_contiguous()),
                    "min": float(frames.min()),
                    "max": float(frames.max()),
                }
                submitted = False
                if frames.is_cuda and frames.dim() == 5:
                    import time as _time

                    import blazar

                    started = _time.perf_counter()
                    batched = frames if frames.is_contiguous() else frames.contiguous()
                    record["contiguous_ms"] = round((_time.perf_counter() - started) * 1000, 2)

                    # One encoder per process. Creating a session per request
                    # costs 300-600 ms, which swamps the 9-frame encode (~4 ms).
                    if _state.get("encoder") is None:
                        # The sink will not create the directory, and a missing
                        # one fails the whole generation at the encoder open.
                        os.makedirs("/var/tmp/blazar/sglang_live", exist_ok=True)
                        output = f"/var/tmp/blazar/sglang_live/persistent_{os.getpid()}.mp4"
                        _state["encoder"] = blazar.Encoder(
                            calibrations=blazar.Calibrations.load(
                                "/home/james/dev/code/blazar-video/calibrations/default"
                            ),
                            identity={"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"},
                            target="balanced",
                            backend="nvenc",
                            codec="h264",
                            output=output,
                        )
                        _state["output"] = output
                    opened = _time.perf_counter()
                    _state["encoder"].submit(batched, fps=16.0, range="unit01")
                    # Encode and write before returning. The encode is async on
                    # the worker and SGLang SIGKILLs this process at teardown, so
                    # a signal handler cannot be relied on to finish the file:
                    # ~40 ms here buys a decodable fragment even under SIGKILL.
                    _state["encoder"].flush()
                    record["flushed"] = True
                    record["submit_ms"] = round((_time.perf_counter() - opened) * 1000, 2)
                    submitted = True
                record["submitted"] = submitted
                with open(_obs, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            return frames

        DecodingStage.decode = decode
        _state["patched"] = True

    def _close_encoder(*_args):
        """The scheduler is killed by its parent, so the only reliable moment to
        finish the file is a signal handler (atexit does not run on SIGTERM)."""
        encoder = _state.get("encoder")
        if encoder is None:
            return
        try:
            encoder.close()
            with open(_obs, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"stage": "encoder_closed", "pid": os.getpid()}) + "\n")
        except Exception as error:  # noqa: BLE001
            with open(_obs, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"stage": "close_failed", "error": str(error)}) + "\n")

    atexit.register(_close_encoder)
    import signal

    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _close_encoder)
        except (ValueError, OSError):
            pass

    try:
        _patch()
    except Exception as error:  # noqa: BLE001 - a hook must never break serving
        with open("/var/tmp/blazar/swizzle_patch_error.txt", "a", encoding="utf-8") as handle:
            handle.write(f"{type(error).__name__}: {error}\n")
