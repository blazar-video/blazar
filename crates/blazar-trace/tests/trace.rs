// SPDX-License-Identifier: Apache-2.0
//! Behaviour of the instrumentation: what it records, what it does not, and that
//! the trace it writes is well formed.

use blazar_trace::{Counter, Stage, Tracer};
use std::time::Duration;

#[test]
fn a_span_records_one_sample_under_its_own_stage() {
    let t = Tracer::new();
    {
        let _s = t.span(Stage::Convert);
        std::thread::sleep(Duration::from_millis(2));
    }
    let snap = t.snapshot();
    let convert = snap.stage(Stage::Convert).expect("convert recorded");
    assert_eq!(convert.count, 1);
    assert!(convert.min_us >= 1500.0, "min {} us", convert.min_us);
    assert!(
        snap.stage(Stage::Upload).is_none(),
        "an unrelated stage must stay empty"
    );
}

#[test]
fn a_disabled_tracer_records_nothing() {
    let t = Tracer::disabled();
    {
        let _s = t.span(Stage::Convert);
        std::thread::sleep(Duration::from_millis(1));
    }
    t.add_counter(Counter::FramesSubmitted, 1);
    let snap = t.snapshot();
    assert!(snap.stages.is_empty());
    assert!(snap.counters.is_empty());
}

#[test]
fn finishing_a_span_by_hand_records_it_once() {
    let t = Tracer::new();
    t.span(Stage::Queue).finish();
    assert_eq!(t.snapshot().stage(Stage::Queue).expect("queue").count, 1);
}

#[test]
fn counters_accumulate() {
    let t = Tracer::new();
    t.add_counter(Counter::BytesDeviceToHost, 6_291_456);
    t.add_counter(Counter::BytesDeviceToHost, 6_291_456);
    let snap = t.snapshot();
    assert_eq!(snap.counter(Counter::BytesDeviceToHost), Some(12_582_912));
}

#[test]
fn percentiles_come_from_the_retained_samples() {
    let t = Tracer::new();
    // 1..=100 microseconds, so p50 is 50 and p95 is 95 by construction.
    for us in 1..=100u64 {
        t.record(Stage::DeviceToHost, Duration::from_micros(us));
    }
    let s = t.snapshot();
    let d2h = s.stage(Stage::DeviceToHost).expect("d2h");
    assert_eq!(d2h.count, 100);
    assert!((49.0..=52.0).contains(&d2h.p50_us), "p50 {}", d2h.p50_us);
    assert!((94.0..=97.0).contains(&d2h.p95_us), "p95 {}", d2h.p95_us);
    assert!((99.0..=100.0).contains(&d2h.max_us), "max {}", d2h.max_us);
}

#[test]
fn the_span_log_is_bounded() {
    let t = Tracer::with_capacities(8, 10);
    for _ in 0..100 {
        t.record(Stage::Convert, Duration::from_micros(3));
    }
    let snap = t.snapshot();
    assert_eq!(snap.stage(Stage::Convert).expect("convert").count, 100);
    assert_eq!(snap.spans_recorded, 10, "the log keeps the newest 10");
    assert_eq!(snap.spans_dropped, 90);
}

#[test]
fn the_sample_window_is_bounded_but_the_count_is_not() {
    let t = Tracer::with_capacities(8, 16);
    for _ in 0..100 {
        t.record(Stage::Convert, Duration::from_micros(3));
    }
    let snap = t.snapshot();
    let s = snap.stage(Stage::Convert).expect("convert");
    assert_eq!(s.count, 100, "every sample is counted");
    assert_eq!(s.samples, 8, "percentiles use the retained window only");
}

#[test]
fn the_chrome_trace_parses_and_describes_each_span() {
    let t = Tracer::new();
    t.record(Stage::Convert, Duration::from_micros(700));
    t.record(Stage::Upload, Duration::from_micros(200));

    let dir = std::env::temp_dir().join(format!("blazar-trace-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("trace.json");
    t.write_chrome(&path).expect("written");

    let text = std::fs::read_to_string(&path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).expect("valid json");
    let events = v["traceEvents"].as_array().expect("traceEvents array");
    assert_eq!(events.len(), 2);
    assert_eq!(events[0]["name"], "convert");
    assert_eq!(events[0]["ph"], "X");
    assert!((events[0]["dur"].as_f64().unwrap() - 700.0).abs() < 50.0);
    assert_eq!(events[1]["name"], "upload");
    assert!(events[0]["ts"].as_f64().unwrap() <= events[1]["ts"].as_f64().unwrap());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn every_stage_has_a_distinct_name() {
    let mut names: Vec<&str> = Stage::ALL.iter().map(|s| s.as_str()).collect();
    names.sort_unstable();
    let n = names.len();
    names.dedup();
    assert_eq!(names.len(), n, "stage names collide");
}
