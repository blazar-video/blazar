// SPDX-License-Identifier: Apache-2.0
//! What the instrumentation costs.
//!
//! Published rather than assumed: a tracing layer that quietly doubles a 1 ms frame
//! would invalidate every number taken with it on.

use blazar_trace::{Counter, Stage, Tracer};
use std::hint::black_box;
use std::time::Instant;

fn ns_per(reps: u32, mut f: impl FnMut()) -> f64 {
    let t0 = Instant::now();
    for _ in 0..reps {
        f();
    }
    t0.elapsed().as_secs_f64() * 1e9 / f64::from(reps)
}

fn main() {
    let reps = 1_000_000;

    let empty = ns_per(reps, || black_box(()));

    let off = Tracer::disabled();
    let disabled = ns_per(reps, || {
        drop(black_box(off.span(Stage::Convert)));
    });

    let on = Tracer::with_capacities(8192, 65_536);
    let enabled = ns_per(200_000, || {
        drop(black_box(on.span(Stage::Convert)));
    });

    let counter = ns_per(reps, || {
        off.add_counter(Counter::BytesDeviceToHost, 1);
    });

    println!("per span/record, on this machine:");
    println!("  empty loop        {empty:6.2} ns");
    println!(
        "  disabled span     {disabled:6.2} ns   (+{:.2} ns)",
        disabled - empty
    );
    println!("  enabled span      {enabled:6.2} ns");
    println!("  disabled counter  {counter:6.2} ns");
    println!();
    println!(
        "an enabled span is {:.2}% of a 1 ms frame; six per frame is {:.2} ms",
        enabled / 1e6 * 100.0,
        enabled * 6.0 / 1e6
    );

    // The same measurement through the 50 ms of frames one 21-frame clip produces.
    let snap = on.snapshot();
    if let Some(s) = snap.stage(Stage::Convert) {
        println!(
            "snapshot: {} spans, mean {:.3} us (this number includes the mutex and Instant::now)",
            s.count, s.mean_us
        );
    }
}
