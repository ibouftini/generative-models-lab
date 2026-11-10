"""Compare a rerun with the published results, file by file.

usage: python -m lab.compare <results folder of the rerun>

Prints one row per results file: how many numbers it holds, how many came out identical, and the largest relative
difference with the number where it occurs. Wall-clock times are left out.
"""
import json
import sys
from pathlib import Path

from . import ROOT

SKIP = {"minutes", "environment", "config", "script", "seed"}


def leaves(node, path=""):
    """Every number in a JSON tree, as (path, value)."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key not in SKIP:
                yield from leaves(value, f"{path}/{key}" if path else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from leaves(value, f"{path}/{i}")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield path, node


def compare(published, rerun):
    a, b = dict(leaves(json.loads(published.read_text()))), dict(leaves(json.loads(rerun.read_text())))
    assert a.keys() == b.keys(), f"{published.name}: the two files hold different numbers"
    worst, where, same = 0.0, "", 0
    for key in a:
        same += a[key] == b[key]
        relative = abs(a[key] - b[key]) / max(abs(a[key]), 1e-12)
        if relative > worst:
            worst, where = relative, key
    return len(a), same, worst, where


def main(folder):
    print("| File | Numbers | Identical | Largest relative difference | Where |\n|---|---:|---:|---:|---|")
    published = ROOT / "results"
    for path in sorted(published.rglob("*.json")):
        name = path.relative_to(published)
        if not (folder / name).exists():
            print(f"| {name} | not rerun | | | |")
            continue
        n, same, worst, where = compare(path, folder / name)
        print(f"| {name} | {n} | {same} | {worst:.2%} | {where} |")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
