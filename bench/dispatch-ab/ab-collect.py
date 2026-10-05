#!/usr/bin/env python3
"""Paired-difference table for the dispatch A/B.

A run's value is the mean over ranks of elastic.py's dispatch+combine avg_t and bandwidth, plus the Kineto
dispatch and combine kernel times when present. For each mode x NVLink cell, rep r pairs the runtime build
with the fixed build; the table reports the median over reps of the fixed-vs-runtime percentage change and
its median absolute deviation (MAD). Negative latency change means the fixed (template-like) build is faster.

usage: ab-collect.py <results dir>
"""
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

BW = re.compile(
    r"\[rank (\d+)\] Dispatch \+ combine bandwidth: ([\d.]+) GB/s, avg_t=([\d.]+) us"
)
KIN = re.compile(
    r"\[rank (\d+)\] Dispatch bandwidth: ([\d.]+) GB/s, avg_t=([\d.]+) us \| "
    r"Combine bandwidth: ([\d.]+) GB/s, avg_t=([\d.]+) us"
)
TAG = re.compile(r"ab-(direct|proxy)-(nvl|nonvl)-(runtime|fixdirect|fixproxy)-r(\d+)$")
METRICS = ("e2e_us", "e2e_gbps", "disp_us", "comb_us")


def load(run_dir):
    rc = run_dir / "rc.txt"
    if not rc.exists() or any(
        not line.strip().endswith("rc=0") for line in rc.read_text().splitlines()
    ):
        return None
    text = "".join(p.read_text(errors="replace") for p in run_dir.glob("*.log"))
    e2e = {int(r): (float(b), float(t)) for r, b, t in BW.findall(text)}
    kin = {int(m[0]): (float(m[2]), float(m[4])) for m in KIN.findall(text)}
    if not e2e:
        return None
    out = {
        "e2e_us": st.mean(t for _, t in e2e.values()),
        "e2e_gbps": st.mean(b for b, _ in e2e.values()),
    }
    if kin:
        out["disp_us"] = st.mean(d for d, _ in kin.values())
        out["comb_us"] = st.mean(c for _, c in kin.values())
    return out


def main():
    root = Path(sys.argv[1])
    runs = defaultdict(dict)  # (mode, nvl) -> {(variant, rep): values}
    missing = []
    for d in sorted(root.iterdir()):
        m = TAG.match(d.name)
        if not m:
            continue
        mode, nvl, variant, rep = m.groups()
        vals = load(d)
        if vals is None:
            missing.append(d.name)
            continue
        runs[(mode, nvl)][
            ("fixed" if variant.startswith("fix") else "runtime", int(rep))
        ] = vals

    print(
        "| mode | NVLink | pairs | metric | runtime median | fixed median | paired change (median) | MAD |"
    )
    print("|---|---|---:|---|---:|---:|---:|---:|")
    for (mode, nvl), cell in sorted(runs.items()):
        reps = sorted(
            {r for _, r in cell if ("runtime", r) in cell and ("fixed", r) in cell}
        )
        for metric in METRICS:
            pairs = [
                (cell[("runtime", r)][metric], cell[("fixed", r)][metric])
                for r in reps
                if metric in cell[("runtime", r)] and metric in cell[("fixed", r)]
            ]
            if not pairs:
                continue
            changes = [100.0 * (f - a) / a for a, f in pairs]
            med = st.median(changes)
            mad = st.median(abs(c - med) for c in changes)
            print(
                f"| {mode} | {nvl} | {len(pairs)} | {metric} | {st.median(a for a, _ in pairs):.3f} | "
                f"{st.median(f for _, f in pairs):.3f} | {med:+.3f}% | {mad:.3f}% |"
            )
    if missing:
        print(f"\nexcluded (failed or no output): {', '.join(missing)}")


if __name__ == "__main__":
    main()
