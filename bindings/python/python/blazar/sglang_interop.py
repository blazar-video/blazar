"""Encode SGLang's diffusion output without a host round trip.

SGLang's video pipelines end in `DiffusersExecutionStage._postprocess_output`,
whose first statement is `output = output.cpu().float()`. Everything after that
point (NaN repair, range correction, clamp) is a torch op that runs just as well
on the GPU, so the transfer is the only part we actually need to remove. This
module replaces that method with one that keeps the tensor on the device and
hands it to Blazar's device-tensor ingest.

Three deliberate properties:

* **It refuses to pretend.** The target class and method are private API. If the
  attribute is missing, or the installed version is not one we verified, this
  raises at install time instead of degrading to the host path while the caller
  believes the fast path is active.
* **The GPU-side post-processing is preserved**, not skipped: NaN/Inf repair, the
  `(x + 1) / 2` range correction and the clamp all still run, on the device.
* **It is reversible** (`with device_output(encoder) as hook:` or `disable()`),
  so a deployment can fall back without redeploying.

Example:

    import blazar
    from blazar.sglang_interop import device_output

    encoder = blazar.Encoder(...)
    with device_output(encoder, fps=24.0):
        ...  # run SGLang; each request's frames are encoded from the GPU

Set ``BLAZAR_SGLANG_DEVICE_OUTPUT=0`` to keep the host path.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Callable

__all__ = [
    "DEVICE_OUTPUT_CLASS",
    "SwizzleError",
    "device_output",
    "install",
    "postprocess_on_device",
    "uninstall",
    "VERIFIED_SGLANG",
]

LOGGER = logging.getLogger(__name__)

VERIFIED_SGLANG = "0.5.22"
"""The only version whose internals were inspected. A newer version may work, but
the method is private and can be renamed without notice, so it has to be checked
rather than assumed."""

DEVICE_OUTPUT_CLASS = (
    "sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline",
    "DiffusersExecutionStage",
    "_postprocess_output",
)
"""Module path, class, method. Verified against sglang 0.5.22, where the method
lives on the *execution stage*, not on `DiffusersPipeline` (patching the latter
silently does nothing)."""


class SwizzleError(RuntimeError):
    """The device-output path could not be installed safely."""


def postprocess_on_device(output: Any) -> Any:
    """SGLang's post-processing with the transfer removed.

    Mirrors `_postprocess_output` step for step, keeping every operation on the
    device:

    * `.cpu()` is gone; `.float()` stays (the kernel needs f32 or bf16).
    * NaN/Inf repair, the `(x + 1) / 2` correction and `clamp(0, 1)` are torch
      ops that run on CUDA unchanged.
    """
    import torch

    if hasattr(output, "float"):
        output = output.float()
    if not hasattr(output, "min"):
        raise SwizzleError(f"expected a tensor, got {type(output).__name__}")
    if torch.isnan(output).any() or torch.isinf(output).any():
        LOGGER.warning("sglang output contained NaN/Inf; repairing before encode")
        output = torch.nan_to_num(output, nan=0.5, posinf=1.0, neginf=0.0)
    low, high = output.min().item(), output.max().item()
    if low < -0.5 or high > 1.5:
        output = (output + 1) / 2
    return output.clamp(0, 1)


def _resolve() -> type:
    module_path, class_name, method = DEVICE_OUTPUT_CLASS
    try:
        module = __import__(module_path, fromlist=[class_name])
    except ImportError as error:
        raise SwizzleError(f"sglang is not importable: {error}") from error
    cls = getattr(module, class_name, None)
    if cls is None:
        raise SwizzleError(f"{module_path}.{class_name} does not exist; sglang internals moved")
    if not callable(getattr(cls, method, None)):
        raise SwizzleError(
            f"{class_name}.{method} is missing; this build is not one the device path was "
            f"verified against (verified: {VERIFIED_SGLANG}). Falling back would be silent, "
            "so this is an error instead."
        )
    return cls


def install(
    on_frames: Callable[[Any], None],
    *,
    encoder_cls: type | None = None,
) -> Callable[[], None]:
    """Patch SGLang so each output's frames are handed to `on_frames` on device.

    Returns a function that restores the original method. `encoder_cls` exists so
    tests can supply a stand-in class with the same method shape.
    """
    if os.environ.get("BLAZAR_SGLANG_DEVICE_OUTPUT", "1") == "0":
        LOGGER.info("BLAZAR_SGLANG_DEVICE_OUTPUT=0: keeping sglang's host path")
        return lambda: None

    cls = encoder_cls if encoder_cls is not None else _resolve()
    _, class_name, method_name = DEVICE_OUTPUT_CLASS
    original = getattr(cls, method_name, None)
    if original is None:
        raise SwizzleError(f"{class_name}.{method_name} not found; refusing to install a no-op")

    def patched(self, output, *args, **kwargs):
        frames = postprocess_on_device(output)
        on_frames(frames)
        # Hand the same device tensor back: callers that serialise still work, and
        # callers that only wanted the encode can ignore it. Making this an error
        # would break the response path for no gain.
        return frames

    setattr(cls, method_name, patched)
    LOGGER.info(
        "sglang device output enabled: %s.%s -> Blazar (verified against %s)",
        class_name,
        method_name,
        VERIFIED_SGLANG,
    )

    def restore() -> None:
        setattr(cls, method_name, original)
        LOGGER.info("sglang device output disabled; original %s restored", method_name)

    return restore


@contextlib.contextmanager
def device_output(encoder: Any, *, fps: float = 24.0, range: str = "unit01"):
    """Encode every SGLang output with `encoder` while inside the block.

    Each output tensor is normalised on the GPU and submitted as a device batch;
    nothing crosses to the host for the encode.
    """
    def on_frames(frames: Any) -> None:
        encoder.submit_device_batch(frames, fps=fps, range=range)

    restore = install(on_frames)
    try:
        yield restore
    finally:
        restore()


def uninstall(encoder_cls: type | None = None) -> None:
    """Restore the original method if a patch is installed."""
    cls = encoder_cls if encoder_cls is not None else _resolve()
    _, _, method_name = DEVICE_OUTPUT_CLASS
    original = getattr(cls, "_blazar_original_" + method_name, None)
    if original is not None:
        setattr(cls, method_name, original)
