import json, os, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import report


def write(path, header, rows):
    with open(path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


def row(name, p50, n=100):
    return {"kind": "row", "row": name, "p50_ms": p50, "p95_ms": p50 * 1.2, "fps": 1000 / p50, "n": n}


def test_table_and_regressions():
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "base.jsonl")
        run = os.path.join(d, "run.jsonl")
        header = {"kind": "header", "provenance": {"git_rev": "abc1234", "blazar_version": "0.1.2",
                                                   "gpu": "x", "os": "y", "wsl": False}}
        write(base, header, [row("a", 2.00), row("b", 5.00)])
        write(run, header, [row("a", 2.02), row("b", 5.60)])
        bheader, brows = report.load(run)
        assert bheader["provenance"]["git_rev"] == "abc1234"
        md = report.table(brows)
        assert "| a | 2.02 |" in md
        regressions = report.compare({r["row"]: r for r in report.load(base)[1]}, brows)
        assert regressions == ["b: 5.00 -> 5.60 ms (+12%)"], regressions


def test_threshold_ignores_noise_and_small_samples():
    base = [row("a", 2.00), row("c", 2.00, n=10)]
    run = [row("a", 2.05), row("c", 3.00, n=10)]
    assert report.compare({r["row"]: r for r in base}, run) == []


def test_main_exits_nonzero_on_regression():
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "base.jsonl")
        run = os.path.join(d, "run.jsonl")
        header = {"kind": "header", "provenance": {"git_rev": "abc", "blazar_version": "0", "gpu": None, "os": "y", "wsl": False}}
        write(base, header, [row("a", 2.00)])
        write(run, header, [row("a", 2.50)])
        assert report.main([run, "--baseline", base]) == 1
        assert report.main([run]) == 0
