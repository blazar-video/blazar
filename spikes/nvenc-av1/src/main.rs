//! NVENC-AV1 hardware spike on the RTX 4080 (mosk).
//!
//! Answers, on real hardware:
//!   1. Does nvidia-video-codec-sdk + cudarc build against CUDA 13.1?
//!   2. Can we open a CUDA context + NVENC-AV1 session on the 4080?
//!   3. Can we encode synthetic frames and get an AV1 bitstream back?
//!
//! Not production code — a hardware capability probe for the Blazar NVENC
//! backend design.

use std::sync::Arc;

use cudarc::driver::CudaContext;
use nvidia_video_codec_sdk::{
    sys::nvEncodeAPI::{
        NV_ENC_BUFFER_FORMAT, NV_ENC_CODEC_AV1_GUID, NV_ENC_PRESET_P4_GUID,
        NV_ENC_TUNING_INFO,
    },
    Encoder, EncodePictureParams, EncoderInitParams,
};

const WIDTH: u32 = 832;
const HEIGHT: u32 = 480;
const FRAMES: u64 = 30;

fn main() {
    println!("== NVENC-AV1 spike: {WIDTH}x{HEIGHT}, {FRAMES} frames ==");

    // 1. CUDA context on device 0 (the 4080).
    let cuda_ctx: Arc<CudaContext> = CudaContext::new(0).expect("CUDA context");
    println!("[ok] CUDA context created on device 0");

    // 2. NVENC encoder bound to CUDA.
    let encoder = Encoder::initialize_with_cuda(cuda_ctx).expect("NVENC init");
    println!("[ok] NVENC encoder initialized");

    // Capability check: is AV1 in the supported encode GUIDs?
    let guids = encoder.get_encode_guids().expect("encode guids");
    let av1 = guids.iter().any(|g| *g == NV_ENC_CODEC_AV1_GUID);
    println!("[info] encode GUIDs: {} total, AV1 supported = {av1}", guids.len());
    assert!(av1, "this GPU does not advertise AV1 NVENC");

    // Input formats for AV1.
    let fmts = encoder
        .get_supported_input_formats(NV_ENC_CODEC_AV1_GUID)
        .expect("input formats");
    println!("[info] AV1 input formats: {fmts:?}");

    // 3. Preset config (P4 = balanced quality/speed), high-quality tuning.
    let mut preset_config = encoder
        .get_preset_config(
            NV_ENC_CODEC_AV1_GUID,
            NV_ENC_PRESET_P4_GUID,
            NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_HIGH_QUALITY,
        )
        .expect("preset config");

    // 4. Start a session. ARGB input: NVENC converts RGB->YUV internally, which
    //    lets the spike skip the preprocess stage and isolate the encoder.
    let buffer_format = NV_ENC_BUFFER_FORMAT::NV_ENC_BUFFER_FORMAT_ARGB;
    let mut init = EncoderInitParams::new(NV_ENC_CODEC_AV1_GUID, WIDTH, HEIGHT);
    init.preset_guid(NV_ENC_PRESET_P4_GUID)
        .tuning_info(NV_ENC_TUNING_INFO::NV_ENC_TUNING_INFO_HIGH_QUALITY)
        .framerate(24, 1)
        .encode_config(&mut preset_config.presetCfg)
        // Let NVENC decide picture types (I/P/B). Without this, the caller must
        // specify a valid pictureType on every frame or NVENC rejects it. With
        // it, NVENC runs its async pipeline and may return NeedMoreInput before
        // emitting — the async-encode contract (architecture §4) we handle below.
        .enable_picture_type_decision();

    let session = encoder
        .start_session(buffer_format, init)
        .expect("start session");
    println!("[ok] NVENC-AV1 session started (ARGB in, P4/HQ)");

    let mut input = session.create_input_buffer().expect("input buffer");
    let mut output = session.create_output_bitstream().expect("output bitstream");

    use nvidia_video_codec_sdk::sys::nvEncodeAPI::NV_ENC_PIC_TYPE;

    let mut total_bytes = 0usize;
    let mut keyframes = 0usize;
    let mut emitted = 0usize;

    // Read whatever bitstream is currently ready and tally it.
    let mut drain_ready = |output: &mut nvidia_video_codec_sdk::Bitstream| {
        let lock = output.lock().expect("lock output");
        let data = lock.data();
        if !data.is_empty() {
            total_bytes += data.len();
            emitted += 1;
            if matches!(
                lock.picture_type(),
                NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_IDR | NV_ENC_PIC_TYPE::NV_ENC_PIC_TYPE_I
            ) {
                keyframes += 1;
            }
        }
    };

    for i in 0..FRAMES {
        // Synthetic ARGB frame: a moving gradient so successive frames differ
        // (gives the encoder real inter-frame content, not a static image).
        let frame = synth_argb(WIDTH, HEIGHT, i);
        unsafe {
            input.lock().expect("lock input").write(&frame);
        }
        match session.encode_picture(
            &mut input,
            &mut output,
            EncodePictureParams { input_timestamp: i, ..Default::default() },
        ) {
            Ok(()) => drain_ready(&mut output),
            // NVENC with lookahead/B-frames buffers frames and asks for more
            // before emitting. Not an error — this IS the async-encode contract
            // (architecture §4 lookahead). Keep feeding; output arrives later.
            Err(e) if e.kind() == nvidia_video_codec_sdk::ErrorKind::NeedMoreInput => {
                // nothing ready yet
            }
            Err(e) => panic!("encode_picture: {e:?}"),
        }
    }

    // Flush the pipeline: EOS, then drain remaining buffered frames.
    session.end_of_stream().expect("eos");
    drain_ready(&mut output);

    println!("[ok] encoded {FRAMES} frames, emitted {emitted} bitstream outputs");
    println!("[result] total bitstream bytes = {total_bytes}");
    println!("[result] keyframes seen = {keyframes}");
    assert!(total_bytes > 0, "no bitstream produced");
    println!("== SPIKE PASS: NVENC-AV1 works on this GPU ==");
}

/// A moving diagonal gradient in ARGB8888 (A,R,G,B per pixel), packed as the
/// NVENC ARGB format expects (little-endian B,G,R,A byte order).
fn synth_argb(w: u32, h: u32, frame: u64) -> Vec<u8> {
    let mut buf = vec![0u8; (w * h * 4) as usize];
    let shift = (frame * 8) as u32;
    for y in 0..h {
        for x in 0..w {
            let r = ((x + shift) & 0xff) as u8;
            let g = ((y + shift) & 0xff) as u8;
            let b = ((x + y) & 0xff) as u8;
            let o = ((y * w + x) * 4) as usize;
            // NVENC ARGB is stored as B,G,R,A in memory.
            buf[o] = b;
            buf[o + 1] = g;
            buf[o + 2] = r;
            buf[o + 3] = 0xff;
        }
    }
    buf
}
