"""Render markdown tables from benchmark JSONL.

The README tables are produced here, from the same records CI produces, so the prose
cannot drift from the measurement.
"""
from __future__ import annotations

import argparse, json, sys

def load(path: str) -> tuple[dict, list[dict]]:
    header, rows = None, []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["kind"] == "header":
                header = rec
            else:
                rows.append(rec)
    return header, rows

def compare(base: dict[str, dict], run: list[dict], threshold: float = 0.05) -> list[str]:
    """Rows whose median moved by more than `threshold`, as report lines."""
    out = []
    for rec in run:
        b = base.get(rec["row"])
        if not b:
            continue
        delta = (rec["p50_ms"] - b["p50_ms"]) / b["p50_ms"]
        if abs(delta) > threshold and rec["n"] >= 50 and b["n"] >= 50:
            out.append(f"{rec['row']}: {b['p50_ms']:.2f} -> {rec['p50_ms']:.2f} ms ({delta:+.0%})")
    return out

def table(rows: list[dict]) -> str:
    lines = [
        "| row | latency p50 | latency p95 | throughput | fps | n |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("p50_ms") is None:
            lines.append(
                f"| {r['row']} | not measurable | - "
                f"| {r['throughput_ms']:.2f} ms | {r['throughput_fps']:.1f} | {r['n']} |"
            )
        else:
            lines.append(
                f"| {r['row']} | {r['p50_ms']:.2f} ms | {r['p95_ms']:.2f} ms "
                f"| {r['throughput_ms']:.2f} ms | {r['throughput_fps']:.1f} | {r['n']} |"
            )
    return "\n".join(lines)

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--baseline")
    ap.add_argument("--threshold", type=float, default=0.05)
    a = ap.parse_args(argv)
    header, rows = load(a.run)
    if header:
        p = header["provenance"]
        print(f"<!-- {p['git_rev'][:8]} {p['blazar_version']} {p['gpu']} {p['os']} wsl={p['wsl']} -->")
    print(table(rows))
    if a.baseline:
        bheader, brows = load(a.baseline)
        del bheader
        regressions = compare({r["row"]: r for r in brows}, rows, a.threshold)
        if regressions:
            print("\nregressions over {:.0%}:".format(a.threshold))
            for line in regressions:
                print(f"- {line}")
            return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
