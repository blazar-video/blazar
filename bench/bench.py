"""Run benchmark scenarios and emit JSONL, one record per row.

The record carries its own provenance so a number can never be quoted without the
build, machine and configuration that produced it. Percentiles are over per-frame
wall time with session open excluded.
"""
from __future__ import annotations

import argparse, json, os, platform, statistics, subprocess, sys, time, tomllib

def provenance() -> dict:
    def sh(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    gpu = sh("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader")
    return {
        "git_rev": sh("git", "rev-parse", "HEAD"),
        "git_dirty": bool(sh("git", "status", "--porcelain")),
        "blazar_version": _version(),
        "cpu": platform.processor() or platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "gpu": gpu.split(", ")[0] if gpu else None,
        "driver": gpu.split(", ")[1] if ", " in gpu else None,
        "wsl": "microsoft" in platform.release().lower(),
        "python": platform.python_version(),
    }

def _version() -> str:
    try:
        import blazar
        return blazar.__version__
    except Exception:
        return "unimportable"

def stats(samples: list[float]) -> dict:
    s = sorted(samples)
    def pct(q: float) -> float:
        return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]
    return {
        "n": len(s),
        "min_ms": round(s[0], 3),
        "p50_ms": round(statistics.median(s), 3),
        "p95_ms": round(pct(0.95), 3),
        "max_ms": round(s[-1], 3),
        "fps": round(1000.0 / s[0], 1),
    }

def run_row(row: dict, cfg: dict) -> dict:
    """Measure one row. Import is local so the schema can be tested without a GPU."""
    import numpy as np
    import torch
    import blazar

    # Without this the record's trace is empty and nothing can be attributed.
    blazar.enable_tracing()

    w, h = int(row["width"]), int(row["height"])
    n = int(cfg["corpus"]["frames"])
    dev = (torch.rand((1, 3, n, h, w), device="cuda", dtype=torch.float32) * 2 - 1).bfloat16()

    enc = blazar.Encoder(
        calibrations=blazar.Calibrations.load(cfg["calibrations"]),
        identity={"family": "wan", "version": "2.1", "quant": "u8", "size": f"{h}p"},
        target=row.get("target", "balanced"), backend=row["backend"], codec=row["codec"],
        output=None, request_id=row["label"], preprocess=row["preprocess"],
    )
    def payload(i: int):
        """What this row submits. A host row pays its own D2H, in the clock."""
        if row["preprocess"] == "host":
            u8 = ((dev[:, :, i].float() * 0.5 + 0.5) * 255.0).round_().clamp_(0, 255).to(torch.uint8)
            return u8[0].permute(1, 2, 0).contiguous().cpu().numpy()
        return dev[:, :, i:i + 1].contiguous()

    done = 0
    failed = False

    def drain_quiet(quiet_s: float = 0.25, timeout_s: float = 120.0) -> bool:
        """Poll until nothing new has arrived for `quiet_s`.

        Counting data-carrying units against a target does not work: 24 frames
        produced 22 such units, because the tail of the stream is still in flight at
        close. A drain that waits for `frames` data units therefore waits forever and
        reports a deadline as if it were a measurement -- which is exactly what the
        first run did.
        """
        nonlocal done, failed
        last = time.monotonic()
        deadline = last + timeout_s
        while time.monotonic() < deadline:
            units = enc.poll()
            new = [u for u in units if u["data"]]
            if new:
                done += len(new)
                last = time.monotonic()
            if enc.stats.get("failed"):
                failed = True
                return False
            if time.monotonic() - last > quiet_s:
                return True
            time.sleep(0.002)
        return False

    def drain_until(total: int, timeout_s: float = 120.0) -> bool:
        """Wait until `total` encoded units have arrived, or give up.

        Never unbounded. A number of encoded units does not always equal the number
        of submitted frames -- B-frames hold some back, and the memory sink emits
        only units that carry data -- so waiting for an exact target can spin
        forever. The first version of this function did, and hung the run.
        """
        nonlocal done, failed
        deadline = time.monotonic() + timeout_s
        while done < total:
            units = enc.poll()
            done += sum(1 for u in units if u["kind"] == "encoded" and u["data"])
            if enc.stats.get("failed"):
                failed = True
                return False
            if time.monotonic() > deadline:
                return False
        return True

    # Warm-up is outside the measurement, and so is session open.
    enc.submit(payload(0), width=w, height=h, fps=24.0, layout="thwc", range="full_u8")
    drain_until(1)

    # Latency: one frame at a time, drained to completion. Off by default because
    # the memory sink batches: poll() can return nothing for several frames and then
    # release four units at once, so this pass waits out its deadline each time.
    latency = []
    incomplete = 0
    for i in range(1, n if row.get("latency", False) else 1):
        t0 = time.perf_counter()
        enc.submit(payload(i), width=w, height=h, fps=24.0, layout="thwc", range="full_u8")
        if not drain_until(done + 1, timeout_s=0.5):
            incomplete += 1
        latency.append((time.perf_counter() - t0) * 1e3)

    # Throughput: the whole clip submitted, then drained. One figure, because the
    # encoder is the bottleneck rather than any single call.
    frames = n - 1
    t0 = time.perf_counter()
    for i in range(1, n):
        enc.submit(payload(i), width=w, height=h, fps=24.0, layout="thwc", range="full_u8")
    complete = drain_quiet(quiet_s=0.5, timeout_s=300.0)
    # The quiet window is part of the elapsed time, so subtract it: the clip is
    # finished when the last unit lands, not when the poller gives up.
    elapsed = time.perf_counter() - t0
    throughput_ms = (elapsed - (0.5 if complete else 0.0)) * 1e3 / frames

    trace = enc.stats.get("trace", {})
    enc.close()

    return {
        "row": row["label"],
        "backend": row["backend"], "codec": row["codec"], "target": row.get("target", "balanced"),
        "preprocess": row["preprocess"], "width": w, "height": h,
        "source": {"dtype": "bf16", "layout": "bcthw", "resident": "device"},
        "basis": "latency: submit then drain each frame; throughput: whole clip",
        **stats(latency if latency else [throughput_ms]),
        "throughput_complete": complete and not failed and incomplete == 0,
        "incomplete_frames": incomplete,
        "throughput_ms": round(throughput_ms, 3),
        "throughput_fps": round(1000.0 / throughput_ms, 1),
        "trace": trace,
    }

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibrations", default="calibrations/default")
    a = ap.parse_args(argv)

    with open(a.scenario, "rb") as f:
        cfg = tomllib.load(f)
    cfg["calibrations"] = a.calibrations

    header = {"kind": "header", "provenance": provenance(), "scenario": a.scenario}
    with open(a.out, "w") as out:
        out.write(json.dumps(header) + "\n")
        for row in cfg["rows"]:
            rec = run_row(row, cfg)
            out.write(json.dumps({"kind": "row", **rec}) + "\n")
            print(f"{rec['row']:16s} {rec['p50_ms']:7.2f} ms  {rec['fps']:7.1f} fps (n={rec['n']})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
