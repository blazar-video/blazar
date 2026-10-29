"""The acceptance suite: what every commit to main and every release must pass.

Every other test file checks one mechanism. This one checks the claim the
package makes, in the form a user would state it: for each backend the build
claims to have, encode a clip in every dtype and both frame locations, and prove
the result decodes with ffmpeg. A backend that is compiled in but cannot produce
a decodable file is a release blocker, not a warning.

The matrix is derived from `blazar.backends()` rather than hard-coded, so a
build that ships fewer encoders runs fewer cases instead of silently passing.
Conversely `test_wheel_carries_the_expected_backends` names the backends the
distribution is expected to contain, which is what catches a wheel that quietly
lost one.

Run it explicitly. Hardware cases skip when no GPU is present; the software
cases (svtav1, vp9, x264, x265, mock) run anywhere. `ffmpeg`/`ffprobe` are
required to verify the output, and the suite fails rather than skipping if they
are absent on a machine that has an encoder, because "we could not check" is not
"it worked".
"""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

import blazar

WIDTH, HEIGHT, FRAMES, FPS = 320, 240, 16, 24.0
CALIBRATIONS = os.environ.get(
    "BLAZAR_CALIBRATIONS",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "calibrations", "default"),
)
IDENTITY = {"family": "wan", "version": "2.1", "quant": "bf16", "size": "1.3b"}

# Which codecs each backend can produce. This mirrors backend_selection.rs; if
# the two disagree the mismatch shows up here as a decode failure.
CODECS = {
    "nvenc": ("h264", "hevc", "av1"),
    "videotoolbox": ("h264", "hevc"),
    "svtav1": ("av1",),
    "x264": ("h264",),
    "x265": ("hevc",),
    "vp9": ("vp9",),
}

# What a user does with a frame: a uint8 array from a camera, or float32 unit
# range from a model. Both are claimed, so both are tested on every backend.
HOST_DTYPES = (
    pytest.param("u8", id="u8"),
    pytest.param("f32", id="f32"),
)


def available() -> set[str]:
    return set(blazar.backends())


def cases() -> list[tuple[str, str]]:
    """(backend, codec) pairs this build claims."""
    present = available()
    return [
        (backend, codec)
        for backend, codecs in CODECS.items()
        if backend in present
        for codec in codecs
    ]


def tool(name: str) -> str:
    for candidate in (
        os.environ.get(name.upper()),
        shutil.which(name),
        os.path.expanduser(f"~/.local/bin/{name}"),
        os.path.expanduser(f"~/.local/ffmpeg-nvenc/bin/{name}"),
    ):
        if candidate and os.path.exists(candidate):
            return candidate
    pytest.fail(f"{name} is required to verify encoded output and was not found")


def host_frames(dtype: str) -> list[np.ndarray]:
    """Moving content, so the encoder codes something rather than flat frames."""
    out = []
    for index in range(FRAMES):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :, 0] = (np.arange(WIDTH) + index * 8)[None, :] % 256
        frame[:, :, 1] = (np.arange(HEIGHT)[:, None] + index * 4) % 256
        frame[:, :, 2] = 64 + index
        out.append(frame if dtype == "u8" else frame.astype(np.float32) / 255.0)
    return out


def torch_device_frames(dtype: str):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("device tensors need CUDA")
    # (1, 3, T, H, W), matching the layout a diffusion model produces.
    base = np.stack([frame for frame in host_frames("u8")], axis=0)
    batch = np.transpose(base, (3, 0, 1, 2))[None]  # (1,3,T,H,W)
    tensor = torch.from_numpy(batch).cuda()
    # Contiguous is a documented requirement of submit(), and a transposed view
    # is exactly what a model that hands over a batch produces.
    if dtype == "bf16":
        return (tensor.to(torch.bfloat16) / torch.tensor(255.0, dtype=torch.bfloat16)).contiguous()
    return (tensor.to(torch.float32) / 255.0).contiguous()


# What the driver says when the build knows about NVENC but the machine has no
# driver: a skip on a hosted runner, a failure on a GPU box. Anything else
# propagates, so a broken session cannot hide behind a skip.
UNAVAILABLE = ("could not load", "no nvidia driver", "nvenc is unavailable", "noencode")


def encoder(backend: str, codec: str, output, request_id: str, **kwargs):
    try:
        return _encoder(backend, codec, output, request_id, **kwargs)
    except RuntimeError as exc:
        if any(marker in str(exc).lower() for marker in UNAVAILABLE):
            pytest.skip(f"{backend} unavailable on this machine: {exc}")
        raise


def _encoder(backend: str, codec: str, output, request_id: str, **kwargs):
    return blazar.Encoder(
        calibrations=blazar.Calibrations.load(CALIBRATIONS),
        identity=IDENTITY,
        target="balanced",
        backend=backend,
        codec=codec,
        output=output,
        request_id=request_id,
        **kwargs,
    )


def decode_check(path: str) -> dict:
    """Assert ffprobe sees one video stream and ffmpeg decodes it without error."""
    probe = subprocess.run(
        [
            tool("ffprobe"), "-v", "error", "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,nb_read_frames",
            "-of", "default=nw=1",
            str(path),
        ],
        capture_output=True, text=True,
    )
    assert probe.returncode == 0, probe.stderr
    fields = dict(
        line.split("=", 1) for line in probe.stdout.splitlines() if "=" in line
    )
    decode = subprocess.run(
        [tool("ffmpeg"), "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True,
    )
    assert decode.returncode == 0, decode.stderr
    assert decode.stderr.strip() == "", decode.stderr
    return fields


@pytest.mark.acceptance
@pytest.mark.parametrize("dtype", HOST_DTYPES)
@pytest.mark.parametrize("backend,codec", cases(), ids=[f"{b}-{c}" for b, c in cases()])
def test_host_frames_decode(tmp_path, backend, codec, dtype):
    """A numpy frame in either supported dtype produces a decodable file."""
    path = tmp_path / f"{backend}-{codec}-{dtype}.mp4"
    enc = encoder(backend, codec, str(path), f"accept-{backend}-{codec}-{dtype}")
    frame_range = "full_u8" if dtype == "u8" else "unit01"
    for frame in host_frames(dtype):
        enc.submit(frame, width=WIDTH, height=HEIGHT, fps=FPS, layout="thwc", range=frame_range)
    enc.close()

    fields = decode_check(str(path))
    assert fields["width"] == str(WIDTH) and fields["height"] == str(HEIGHT), fields
    assert int(fields["nb_read_frames"]) == FRAMES, fields


@pytest.mark.acceptance
@pytest.mark.parametrize("dtype", [pytest.param("bf16", id="bf16"), pytest.param("f32", id="f32")])
@pytest.mark.parametrize(
    "backend,codec", [(b, c) for b, c in cases() if b == "nvenc"], ids=lambda v: str(v)
)
def test_device_frames_decode(tmp_path, backend, codec, dtype):
    """A CUDA tensor is encoded in place and the result decodes."""
    tensor = torch_device_frames(dtype)
    path = tmp_path / f"{backend}-{codec}-device-{dtype}.mp4"
    enc = encoder(backend, codec, str(path), f"accept-{backend}-{codec}-dev-{dtype}", layout="bcthw")
    enc.submit(tensor, fps=FPS)
    enc.close()

    assert enc.residency == "device"
    fields = decode_check(str(path))
    assert int(fields["nb_read_frames"]) == FRAMES, fields


@pytest.mark.acceptance
@pytest.mark.parametrize("placement", ["cpu", "gpu"])
@pytest.mark.parametrize(
    "backend,codec",
    [("nvenc", "h264"), ("nvenc", "hevc"), ("nvenc", "av1"), ("svtav1", "av1"), ("vp9", "vp9")],
    ids=lambda v: str(v),
)
def test_both_conversion_sites_decode(tmp_path, backend, codec, placement):
    """RGB->NV12 on either side of the bus produces the same decodable class.

    `gpu` is allowed everywhere the kernel exists, including software backends
    (the NV12 comes back over the bus). What is refused is `cpu` for frames that
    are already on the device, asserted separately below.
    """
    present = available()
    if backend not in present:
        pytest.skip(f"{backend} not built")

    path = tmp_path / f"{backend}-{codec}-{placement}.mp4"
    enc = encoder(backend, codec, str(path), f"accept-{backend}-{codec}-{placement}", preprocess=placement)
    for frame in host_frames("u8"):
        enc.submit(frame, width=WIDTH, height=HEIGHT, fps=FPS, layout="thwc", range="full_u8")
    enc.close()

    assert enc.preprocess == placement, enc.preprocess
    fields = decode_check(str(path))
    assert int(fields["nb_read_frames"]) == FRAMES, fields


@pytest.mark.acceptance
def test_cpu_placement_refuses_device_frames(tmp_path):
    """The one combination that cannot work: device frames, host conversion."""
    if "nvenc" not in available():
        pytest.skip("nvenc not built")
    tensor = torch_device_frames("bf16")
    enc = encoder(
        "nvenc", "h264", str(tmp_path / "refused.mp4"), "accept-refused",
        layout="bcthw", preprocess="cpu",
    )
    with pytest.raises(RuntimeError):
        enc.submit(tensor, fps=FPS)


@pytest.mark.acceptance
def test_audio_and_video_mux_read_back(tmp_path):
    """Opus + video in one MP4: both streams present, whole file decodes."""
    backend = "nvenc" if "nvenc" in available() else ("svtav1" if "svtav1" in available() else None)
    if backend is None:
        pytest.skip("no video backend to pair with audio")
    codec = "h264" if backend == "nvenc" else "av1"
    path = tmp_path / "av.mp4"
    try:
        enc = encoder(backend, codec, str(path), "accept-audio", audio="opus")
    except (RuntimeError, ValueError) as exc:  # built without the opus feature
        pytest.skip(f"opus audio not built: {exc}")
    for frame in host_frames("u8"):
        enc.submit(frame, width=WIDTH, height=HEIGHT, fps=FPS, layout="thwc", range="full_u8")
    n = 48_000
    wave = 0.5 * np.sin(2 * np.pi * 440.0 * np.arange(n) / 48_000)
    enc.submit_audio(np.repeat(wave.astype(np.float32), 2), sample_rate=48_000)
    enc.close()

    probe = subprocess.run(
        [tool("ffprobe"), "-v", "error", "-show_entries", "stream=codec_type,codec_name",
         "-of", "default=nw=1", str(path)],
        capture_output=True, text=True,
    )
    assert probe.returncode == 0, probe.stderr
    assert "codec_type=audio" in probe.stdout and "codec_type=video" in probe.stdout, probe.stdout
    decode = subprocess.run(
        [tool("ffmpeg"), "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True,
    )
    assert decode.returncode == 0, decode.stderr


@pytest.mark.acceptance
@pytest.mark.parametrize("backend,codec", cases(), ids=[f"{b}-{c}" for b, c in cases()])
def test_memory_sink_returns_units(backend, codec):
    """`output=None` yields encoded bytes for every backend, with no file."""
    enc = encoder(backend, codec, None, f"accept-mem-{backend}-{codec}")
    for frame in host_frames("u8"):
        enc.submit(frame, width=WIDTH, height=HEIGHT, fps=FPS, layout="thwc", range="full_u8")
    enc.close()
    units = [e for e in enc.poll() if e["kind"] == "encoded" and e.get("data")]
    assert units, "no encoded units"
    assert all(e["dts"] is not None for e in units), "dts missing from events"


@pytest.mark.acceptance
def test_backends_report_what_this_wheel_builds():
    """The build's own claim. A wheel that loses an encoder fails here first."""
    present = available()
    expected = {
        "blazar-video": {"nvenc", "svtav1", "vp9"},
        "blazar-video-windows": {"nvenc"},
        "blazar-video-gpl": {"nvenc", "svtav1", "vp9", "x264", "x265"},
        "dev": {"nvenc", "svtav1", "vp9", "x264"},
    }
    name = os.environ.get("BLAZAR_ACCEPTANCE_FLAVOUR", "dev")
    assert name in expected, name
    missing = expected[name] - present
    assert not missing, f"{name} is missing {sorted(missing)}; present: {sorted(present)}"
    if name != "blazar-video-gpl":
        assert "x265" not in present or name == "dev", "an Apache wheel must not carry x265"
