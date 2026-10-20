"""SGLang device-output swizzle.

Tested against a stand-in class with the same shape as the real one, so the suite
does not need a GPU, a model or sglang itself.
"""

import importlib.util
import pathlib

import pytest
import torch


def _load():
    """Load the module by path: the package __init__ imports the compiled
    binding, which this test deliberately does not need (it takes any encoder)."""
    path = pathlib.Path(__file__).resolve().parents[1] / "python" / "blazar" / "sglang_interop.py"
    spec = importlib.util.spec_from_file_location("sglang_interop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


interop = _load()
DEVICE_OUTPUT_CLASS = interop.DEVICE_OUTPUT_CLASS
SwizzleError = interop.SwizzleError
install = interop.install
postprocess_on_device = interop.postprocess_on_device


class Stage:
    """Same shape as sglang's DiffusersExecutionStage method."""

    def _postprocess_output(self, output):
        output = output.cpu().float()
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.nan_to_num(output, nan=0.5, posinf=1.0, neginf=0.0)
        if output.min() < -0.5 or output.max() > 1.5:
            output = (output + 1) / 2
        return output.clamp(0, 1)


def tensors():
    return torch.tensor([[-1.0, 0.0, 1.0]]), torch.tensor([[0.2, 0.4, 0.6]]), torch.tensor([[float("nan"), 0.0, 2.0]])


def test_install_repatches_and_restores():
    original = Stage._postprocess_output
    calls = []
    restore = install(calls.append, encoder_cls=Stage)
    assert Stage._postprocess_output is not original
    Stage()._postprocess_output(tensors()[1])
    assert len(calls) == 1
    restore()
    assert Stage._postprocess_output is original


def test_the_patch_receives_frames_and_keeps_their_values():
    seen = []
    restore = install(seen.append, encoder_cls=Stage)
    try:
        Stage()._postprocess_output(tensors()[1])
    finally:
        restore()
    assert torch.allclose(seen[0], tensors()[1])


def test_postprocessing_matches_what_sglang_does():
    for frame in tensors():
        expected = Stage()._postprocess_output(frame)
        assert torch.allclose(postprocess_on_device(frame), expected, equal_nan=True)


def test_negative_range_is_rescaled_not_clipped():
    frame = torch.tensor([[-1.0, 0.0, 1.0]])
    out = postprocess_on_device(frame)
    assert out.min().item() == pytest.approx(0.0)
    assert out.max().item() == pytest.approx(1.0)


def test_nan_is_repaired_rather_than_encoded():
    out = postprocess_on_device(torch.tensor([[float("nan"), 0.0, 1.0]]))
    assert not torch.isnan(out).any()


def test_missing_method_is_an_error_not_a_silent_noop():
    class Moved:
        pass

    with pytest.raises(SwizzleError):
        install(lambda frames: None, encoder_cls=Moved)


def test_kill_switch_installs_nothing(monkeypatch):
    monkeypatch.setenv("BLAZAR_SGLANG_DEVICE_OUTPUT", "0")
    original = Stage._postprocess_output
    restore = install(lambda frames: None, encoder_cls=Stage)
    assert Stage._postprocess_output is original
    restore()
    assert Stage._postprocess_output is original


def test_the_verified_target_is_the_execution_stage():
    # Patching DiffusersPipeline instead of DiffusersExecutionStage silently does
    # nothing; this pins the class that was actually checked in sglang 0.5.22.
    module, cls, method = DEVICE_OUTPUT_CLASS
    assert cls == "DiffusersExecutionStage"
    assert method == "_postprocess_output"
    assert module.endswith("diffusers_pipeline")
