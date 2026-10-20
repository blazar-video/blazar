# SGLang device-output patch

`sitecustomize.py` installs Blazar's device-output hook inside every process
SGLang spawns. Put this directory first on `PYTHONPATH` and set
`BLAZAR_SWIZZLE=1`; Python imports the module at startup in each interpreter,
which is the seam that works, because SGLang runs its scheduler in a *spawned*
child where a parent-process monkey-patch never arrives.

## What the live run established (sglang 0.5.22, Wan 2.1 T2V 1.3B)

- Wan does **not** use `DiffusersExecutionStage`; that path is for checkpoints
  loaded straight from diffusers. The Wan pipeline composes `DecodingStage`, and
  `DecodingStage.decode` returns the decoded frames still on the device.
  `DecodingStage.forward` is the wrong hook: with an output file requested it
  returns `output=None, output_file_paths=[...]`, because SGLang encodes inside
  `_encode_while_decoding`.
- The tensor arrives as `[1, 3, 9, 320, 480]`, `torch.float32`, `cuda:0`, values
  in `[0, 1]` — exactly the BCTHW layout `submit_device_batch` wants, except that
  it is **not contiguous**. `.contiguous()` costs 0.11-0.13 ms for 9 frames.
- **One encoder per process.** Creating a session per request costs 337-598 ms,
  against roughly 4 ms to encode the 9 frames. Per-request encoders are slower
  than the host path they replace.
- **Close on SIGTERM, not atexit.** SGLang's parent kills the scheduler, and
  atexit does not run on SIGTERM. Without a handler the muxer never writes its
  trailer and the file is 0 bytes with all frames "encoded".

Output from the verified run: `persistent_<pid>.mp4`, 9 frames, 480x320, h264,
192,739 bytes (Blazar CRF 30 balanced), `ffmpeg -xerror` decode clean. SGLang's
own file for the same generation is 171,910 bytes.
