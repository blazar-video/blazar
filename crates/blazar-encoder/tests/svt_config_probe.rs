//! Which configuration fields does the installed SVT-AV1 accept?
//!
//! Run with `--nocapture`: the versions differ in what `set_parameter` validates,
//! and this names the field instead of guessing from a generic error code.

#![cfg(feature = "enc-svtav1")]

use std::ptr;
use blazar_svtav1_sys as sys;

unsafe fn attempt(name: &str, apply: impl FnOnce(&mut sys::EbSvtAv1EncConfiguration)) {
    let mut config: sys::EbSvtAv1EncConfiguration = std::mem::zeroed();
    let mut handle: *mut sys::EbComponentType = ptr::null_mut();
    let rc = sys::enc_init_handle(&mut handle, &mut config);
    if rc != sys::EbErrorType::EB_ErrorNone {
        println!("{name:40} init_handle failed ({rc})");
        return;
    }
    // Baseline that must work everywhere.
    config.source_width = 64;
    config.source_height = 48;
    config.frame_rate_numerator = 24;
    config.frame_rate_denominator = 1;
    config.encoder_color_format = sys::EbColorFormat::EB_YUV420;
    sys::set_rate_control_cqp(&mut config, 32);
    apply(&mut config);
    let rc = sys::svt_av1_enc_set_parameter(handle, &mut config);
    println!(
        "{name:40} {}",
        if rc == sys::EbErrorType::EB_ErrorNone { "ok".to_string() } else { format!("FAILED ({rc})") }
    );
    let _ = sys::svt_av1_enc_deinit(handle);
    let _ = sys::svt_av1_enc_deinit_handle(handle);
}

unsafe fn attempt_untouched(name: &str, mutate: impl FnOnce(&mut sys::EbSvtAv1EncConfiguration)) {
    let mut config: sys::EbSvtAv1EncConfiguration = std::mem::zeroed();
    let mut handle: *mut sys::EbComponentType = ptr::null_mut();
    let rc = sys::enc_init_handle(&mut handle, &mut config);
    if rc != sys::EbErrorType::EB_ErrorNone {
        println!("{name:40} init_handle failed ({rc})");
        return;
    }
    mutate(&mut config);
    let set = sys::svt_av1_enc_set_parameter(handle, &mut config);
    let init = if set == sys::EbErrorType::EB_ErrorNone {
        sys::svt_av1_enc_init(handle)
    } else {
        -1
    };
    println!(
        "{name:40} set={} init={}",
        if set == sys::EbErrorType::EB_ErrorNone { "ok" } else { "FAILED" },
        if init == sys::EbErrorType::EB_ErrorNone { "ok" } else { "skipped/failed" }
    );
    let _ = sys::svt_av1_enc_deinit(handle);
    let _ = sys::svt_av1_enc_deinit_handle(handle);
}

#[test]
fn probe_config_fields() {
    unsafe {
        println!("EB_ErrorNone={} EB_ErrorBadParameter={}", sys::EbErrorType::EB_ErrorNone, sys::EbErrorType::EB_ErrorBadParameter);
        println!("rust sizeof(EbSvtAv1EncConfiguration)={} sizeof(EbBufferHeaderType)={}",
                 std::mem::size_of::<sys::EbSvtAv1EncConfiguration>(),
                 std::mem::size_of::<sys::EbBufferHeaderType>());
        attempt_untouched("untouched", |_| {});
        attempt_untouched("full 2.x default-ish", |c| {
            c.source_width = 64;
            c.source_height = 48;
            c.frame_rate_numerator = 24;
            c.frame_rate_denominator = 1;
            c.encoder_color_format = sys::EbColorFormat::EB_YUV420;
            c.intra_period_length = 8;
            c.enc_mode = 8;
            sys::set_rate_control_cqp(c, 32);
        });
        attempt_untouched("full + random levels3", |c| {
            c.source_width = 64;
            c.source_height = 48;
            c.frame_rate_numerator = 24;
            c.frame_rate_denominator = 1;
            c.encoder_color_format = sys::EbColorFormat::EB_YUV420;
            c.intra_period_length = 8;
            c.enc_mode = 8;
            c.hierarchical_levels = 3;
            sys::set_random_access(c);
            sys::set_rate_control_cqp(c, 32);
        });
        attempt_untouched("full + random levels0", |c| {
            c.source_width = 64;
            c.source_height = 48;
            c.frame_rate_numerator = 24;
            c.frame_rate_denominator = 1;
            c.encoder_color_format = sys::EbColorFormat::EB_YUV420;
            c.intra_period_length = 8;
            c.enc_mode = 8;
            c.hierarchical_levels = 0;
            sys::set_random_access(c);
            sys::set_rate_control_cqp(c, 32);
        });
        attempt_untouched("full, intra=5", |c| {
            c.source_width = 64;
            c.source_height = 48;
            c.frame_rate_numerator = 24;
            c.frame_rate_denominator = 1;
            c.encoder_color_format = sys::EbColorFormat::EB_YUV420;
            c.intra_period_length = 5;
            c.enc_mode = 8;
            c.hierarchical_levels = 0;
            sys::set_random_access(c);
            sys::set_rate_control_cqp(c, 32);
        });
        attempt_untouched("width/height only", |c| { c.source_width = 64; c.source_height = 48; });
        attempt_untouched("format only", |c| { c.encoder_color_format = sys::EbColorFormat::EB_YUV420; });
        attempt_untouched("rc only", |c| sys::set_rate_control_cqp(c, 32));
        attempt("baseline", |_| {});
        attempt("intra_period_length=5", |c| c.intra_period_length = 5);
        attempt("intra_period_length=60", |c| c.intra_period_length = 60);
        attempt("enc_mode=10", |c| c.enc_mode = 10);
        attempt("enc_mode=8", |c| c.enc_mode = 8);
        attempt("pred_structure=random", sys::set_random_access);
        attempt("pred_structure=random+levels0", |c| {
            sys::set_random_access(c);
            c.hierarchical_levels = 0;
        });
        attempt("horiz/qp only", |_| {});
        attempt("fps 24000/1000", |c| {
            c.frame_rate_numerator = 24000;
            c.frame_rate_denominator = 1000;
        });
    }
}
