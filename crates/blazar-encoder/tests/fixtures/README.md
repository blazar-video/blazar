# Test fixtures

## `nvenc_av1_keyframe_head.bin`

First 128 bytes of a real NVENC AV1 keyframe, captured on mosk
(RTX 4080 SUPER, driver 610.57, CUDA 13.1) while probing the backend design.
Input was 832x480 NV12, encoder P4/HQ tuning, picture-type decision enabled,
`gopLength=1000`, `frameIntervalP=1`.

Contents: `OBU_TEMPORAL_DELIMITER` (size 0), `OBU_SEQUENCE_HEADER` (11-byte
payload), then the start of `OBU_FRAME`.

Not synthetic: these are driver-produced bytes. Do not regenerate by hand.
