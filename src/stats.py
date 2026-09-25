"""Test which attributes correlate with the performance metric.

Each row is one data point. For every attribute:
  - choice / yes-no labels: compare mean metric across groups
      two groups -> Welch t-test, three or more -> one-way ANOVA
  - numbers (word count, Jev probabilities and option weights): Spearman
P-values are corrected for multiple comparisons (Benjamini-Hochberg).
Low-confidence Jev answers are kept; they are flagged upstream, never dropped.

Usage: python src/stats.py --input output/tags.csv --output output/results.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
METRIC = "conversion_rate"
ALPHA = 0.05


def benjamini_hochberg(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def group_test(name, labels, y):
    groups = {}
    for lab, v in zip(labels, y):
        groups.setdefault(lab, []).append(v)
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if len(groups) < 2:
        return None
    samples = list(groups.values())
    if len(groups) == 2:
        test, p = "t-test", stats.ttest_ind(*samples, equal_var=False).pvalue
    else:
        test, p = "ANOVA", stats.f_oneway(*samples).pvalue
    means = {k: float(np.mean(v)) for k, v in groups.items()}
    best, worst = max(means, key=means.get), min(means, key=means.get)
    return {
        "attribute": name, "test": test, "p_value": float(p),
        "groups": {k: {"n": len(v), "mean": means[k]} for k, v in groups.items()},
        "effect_pts": 100 * (means[best] - means[worst]),
        "summary": f"{best} vs {worst}",
    }


def correlation_test(name, x, y):
    rho, p = stats.spearmanr(x, y)
    return {"attribute": name, "test": "Spearman", "p_value": float(p),
            "groups": {}, "effect_pts": None, "rho": float(rho),
            "summary": f"rho {rho:+.2f}"}


def run(rows, config):
    y = [float(r[METRIC]) for r in rows]
    results = []
    for a in config["attributes"]:
        n = a["name"]
        if a["computed_by"] == "code":
            results.append(correlation_test(n, [float(r[n]) for r in rows], y))
            continue
        results.append(group_test(n, [r[f"jev_{n}"] for r in rows], y))
        if a["type"] == "boolean":
            results.append(correlation_test(
                f"{n} (probability yes)", [float(r[f"jev_{n}_p"]) for r in rows], y))
        elif config["settings"].get("test_option_weights"):
            # each option's weight as a sliding scale, for blended copy
            for opt in a["options"]:
                x = [json.loads(r[f"jev_{n}_weights"]).get(opt, 0.0) for r in rows]
                if np.std(x) > 0:
                    results.append(correlation_test(f"{n} (weight: {opt})", x, y))
    results = [r for r in results if r]
    for r, adj in zip(results, benjamini_hochberg([r["p_value"] for r in results])):
        r["p_adjusted"] = float(adj)
        r["significant"] = "yes" if adj < ALPHA else "no"
    return results


def write(results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["attribute", "test", "group", "n", "mean_" + METRIC,
                    "effect_pts", "rho", "p_value", "p_adjusted", "significant"])
        for r in results:
            base = [r["attribute"], r["test"]]
            tail = [f"{r['effect_pts']:.2f}" if r["effect_pts"] is not None else "",
                    f"{r['rho']:+.3f}" if "rho" in r else "",
                    f"{r['p_value']:.2e}", f"{r['p_adjusted']:.2e}", r["significant"]]
            if r["groups"]:
                for g, d in sorted(r["groups"].items()):
                    w.writerow(base + [g, d["n"], f"{d['mean']:.4f}"] + tail)
            else:
                w.writerow(base + ["", "", ""] + tail)


def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "output/tags.csv"))
    parser.add_argument("--output", default=str(ROOT / "output/results.csv"))
    parser.add_argument("--config", default=str(ROOT / "attributes.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    rows = list(csv.DictReader(open(args.input)))
    results = run(rows, config)
    write(results, args.output)
    for r in results:
        eff = f"{r['effect_pts']:+.2f} pts" if r["effect_pts"] is not None else f"rho {r['rho']:+.2f}"
        print(f"  {'SIGNIFICANT' if r['significant'] == 'yes' else '           '}  "
              f"{r['attribute']:34} {r['test']:9} {eff:>12}  "
              f"({r['summary']}, adj p={r['p_adjusted']:.1e})")


if __name__ == "__main__":
    main()
