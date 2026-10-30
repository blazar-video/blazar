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

    w, h = int(row["width"]), int(row["height"])
    n = int(cfg["corpus"]["frames"])
    dev = (torch.rand((1, 3, n, h, w), device="cuda", dtype=torch.float32) * 2 - 1).bfloat16()

    enc = blazar.Encoder(
        calibrations=blazar.Calibrations.load(cfg["calibrations"]),
        identity={"family": "wan", "version": "2.1", "quant": "u8", "size": f"{h}p"},
        target="balanced", backend=row["backend"], codec=row["codec"],
        output=None, request_id=row["label"], preprocess=row["preprocess"],
    )
    # Warm-up is not part of the measurement, and neither is session open.
    enc.submit(dev[:, :, 0:1].contiguous(), width=w, height=h, fps=24.0,
               layout="thwc", range="full_u8")
    enc.poll()

    samples = []
    for i in range(1, n):
        frame = dev[:, :, i:i + 1].contiguous()
        t0 = time.perf_counter()
        enc.submit(frame, width=w, height=h, fps=24.0, layout="thwc", range="full_u8")
        enc.poll()
        samples.append((time.perf_counter() - t0) * 1e3)
    trace = enc.stats.get("trace", {})
    enc.close()

    return {
        "row": row["label"],
        "backend": row["backend"], "codec": row["codec"],
        "preprocess": row["preprocess"], "width": w, "height": h,
        "source": {"dtype": "bf16", "layout": "bcthw", "resident": "device"},
        **stats(samples),
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
