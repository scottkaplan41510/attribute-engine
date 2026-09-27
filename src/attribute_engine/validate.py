"""Close the loop: did copy written from the winners actually perform better?

Fill in the metric column of output/creative_log.csv once the new copy has run,
then run this. It compares generated copy (built from the winning attributes)
against the control (built from the losing ones) and against the originals.

Usage: attribute-engine-validate [--log output/creative_log.csv]
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats

from .stats import METRIC


def compare(rows, metric=METRIC):
    groups = {}
    for r in rows:
        value = (r.get(metric) or "").strip()
        if value:
            groups.setdefault(r["source"], []).append(float(value))
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=str(Path.cwd() / "output/creative_log.csv"))
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.log)))
    groups = compare(rows)

    print(f"{METRIC.replace('_', ' ')} by source ({args.log}):")
    for source in ("generated", "control", "original"):
        vals = groups.get(source, [])
        mean = f"{np.mean(vals):.4f}" if vals else "no results yet"
        print(f"  {source:10} {mean:>16}  ({len(vals)} with results)")

    gen, ctl = groups.get("generated", []), groups.get("control", [])
    if len(gen) >= 2 and len(ctl) >= 2:
        p = stats.ttest_ind(gen, ctl, equal_var=False).pvalue
        verdict = ("held up: generated beat the control" if np.mean(gen) > np.mean(ctl) and p < 0.05
                   else "not proven yet: no significant difference")
        print(f"\nWinners vs control: {verdict} (p = {p:.3g}).")
    else:
        print("\nNeed results for at least 2 generated and 2 control pieces to compare.")
    print("This is a correlation check on real results, not proof of cause.")


if __name__ == "__main__":
    main()
