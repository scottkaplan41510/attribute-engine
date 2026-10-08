"""Test which attributes correlate with the performance metric.

Each row is one data point. For every attribute:
  - choice / yes-no labels: compare mean metric across groups
      two groups -> Welch t-test, three or more -> one-way ANOVA
  - numbers (word count, Jev probabilities and option weights): Spearman
P-values are corrected for multiple comparisons (Benjamini-Hochberg).
Low-confidence Jev answers are kept; they are flagged upstream, never dropped.

Usage: python -m attribute_engine.stats --input output/tags.csv --output output/results.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path.cwd()  # data/, output/ and .env are read from where you run it
CONFIG = Path(__file__).resolve().parent / "attributes.yaml"
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


def scorecard(rows, config):
    """One row per attribute value, in plain terms: rows with that value vs
    every other row.

    Choice and yes/no attributes: every value with at least 2 rows on each side
    gets its own row. The correlation is Pearson between "has this value" (1/0)
    and the metric (point-biserial). Every value is tested, not just the best
    one, so two strong values can both show, and a negative correlation means
    the value goes with a lower metric.
    Counted attributes (word count): Spearman, with no value.

    Significance has two gates, so testing many values can't make noise look
    real: the attribute as a whole must differ (t-test or ANOVA across all its
    values, adjusted across attributes) AND the value must differ from the rest
    (adjusted across values). The p-value reported is the larger of the two,
    so it always agrees with "significant". Benjamini-Hochberg throughout.
    Rows are grouped by attribute, strongest attribute first.
    """
    y = np.array([float(r[METRIC]) for r in rows])
    out, attr_p = [], {}
    for a in config["attributes"]:
        n = a["name"]
        empty = {"attribute": n, "value": None, "n_with": None, "mean_with": None,
                 "n_rest": None, "mean_rest": None, "lift": None, "lift_pct": None}
        if a["computed_by"] == "code":
            x = np.array([float(r[n]) for r in rows])
            if np.std(x) == 0:
                continue
            rho, p = stats.spearmanr(x, y)
            attr_p[n] = float(p)
            out.append({**empty, "kind": "number", "correlation": float(rho), "p_value": float(p)})
            continue
        labels = np.array([r[f"jev_{n}"] for r in rows])
        overall = group_test(n, labels, y)
        if overall is None:
            continue
        attr_p[n] = overall["p_value"]
        for v in sorted(set(labels)):
            has = labels == v
            if not 2 <= has.sum() <= len(labels) - 2:
                continue
            r_pb, p = stats.pearsonr(has.astype(float), y)
            mean_with, mean_rest = float(y[has].mean()), float(y[~has].mean())
            out.append({**empty, "kind": "group", "value": str(v),
                        "n_with": int(has.sum()), "mean_with": mean_with,
                        "n_rest": int((~has).sum()), "mean_rest": mean_rest,
                        "lift": mean_with - mean_rest,
                        "lift_pct": (mean_with - mean_rest) / mean_rest if mean_rest else None,
                        "correlation": float(r_pb), "p_value": float(p)})
    if not out:
        return out
    attr_adj = dict(zip(attr_p, benjamini_hochberg(list(attr_p.values()))))
    groups = [r for r in out if r["kind"] == "group"]
    value_adj = benjamini_hochberg([r["p_value"] for r in groups]) if groups else []
    for r, adj in zip(groups, value_adj):
        r["p_value_adjusted"] = float(adj)
    for r in out:
        p = attr_adj[r["attribute"]]
        if r["kind"] == "group":
            p = max(p, r["p_value_adjusted"])
        r["p_adjusted"] = float(p)
        r["significant"] = bool(p < ALPHA)
    strength = {}
    for r in out:
        strength[r["attribute"]] = max(strength.get(r["attribute"], 0), abs(r["correlation"]))
    return sorted(out, key=lambda r: (-strength[r["attribute"]], r["attribute"], -r["correlation"]))


def write_scorecard(card, path):
    fields = ["attribute", "value", "n_with", "mean_with", "n_rest", "mean_rest",
              "lift", "lift_pct", "correlation", "p_value", "p_adjusted", "significant"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(card)


def print_scorecard(card):
    def pct(v):
        return "" if v is None else f"{100 * v:.1f}%"
    print(f"  {'ATTRIBUTE':28} {'VALUE':16} {'WITH IT':>13} {'THE REST':>13} "
          f"{'LIFT':>9} {'CORR':>6} {'ADJ P':>8}  SIGNIFICANT")
    for r in card:
        with_it = f"{pct(r['mean_with'])} ({r['n_with']})" if r["kind"] == "group" else ""
        rest = f"{pct(r['mean_rest'])} ({r['n_rest']})" if r["kind"] == "group" else ""
        lift = f"{100 * r['lift']:+.1f} pts" if r["lift"] is not None else ""
        print(f"  {r['attribute']:28} {r['value'] or '':16} {with_it:>13} {rest:>13} "
              f"{lift:>9} {r['correlation']:+6.2f} {r['p_adjusted']:8.2g}  "
              f"{'yes' if r['significant'] else 'no'}")


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
    parser.add_argument("--config", default=str(CONFIG))
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
