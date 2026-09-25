"""Plain-English report. Every finding is worded as a correlation, never a cause."""

from .stats import METRIC

CAVEAT = ("These are correlations, not causes. A small sample only catches big "
          "effects, so treat anything here as a hypothesis to test.")


def _line(r):
    label = METRIC.replace("_", " ")
    if r["groups"]:
        means = r["groups"]
        best = max(means, key=lambda g: means[g]["mean"])
        worst = min(means, key=lambda g: means[g]["mean"])
        text = (f"{r['attribute']} = {best} correlates with a {r['effect_pts']:.1f} point "
                f"higher {label} than {worst} ({100 * means[best]['mean']:.1f}% vs "
                f"{100 * means[worst]['mean']:.1f}%)")
    else:
        direction = "higher" if r["rho"] > 0 else "lower"
        text = f"more {r['attribute']} correlates with {direction} {label} (rho {r['rho']:+.2f})"
    verdict = "significant" if r["significant"] == "yes" else "not significant"
    return f"- {text}. {verdict.capitalize()} (adjusted p = {r['p_adjusted']:.3g})."


def _headline_rows(results):
    """One line per attribute: the group comparison if there is one."""
    seen, out = set(), []
    for r in results:
        base = r["attribute"].split(" (")[0]
        if base in seen:
            continue
        seen.add(base)
        out.append(r)
    return out


def write_report(round1, round2, found, made, n, path):
    lines = ["# Attribute analysis", "", f"{n} rows analyzed. {CAVEAT}", "",
             "## Round 1: attributes you defined", ""]
    lines += [_line(r) for r in _headline_rows(round1)]
    lines += ["", "## Blind discovery", "",
              "Claude read the copy only, never the metric, and proposed:", ""]
    lines += [f"- {a['name']}: {a['question']}" for a in found["proposed"]]
    lines += ["", "## Round 2: defined plus discovered", ""]
    lines += [_line(r) for r in _headline_rows(round2)]
    if made["new_copy"]:
        lines += ["", "## New copy from the winning attributes", "",
                  "Written by Claude to use: "
                  + ", ".join(f"{w['name']} = {w['value']}" for w in made["winners"]) + ".",
                  "Checked by Jev, and saved to ledger.json with an ID so it can be "
                  "scored once it runs.", ""]
        for rec in made["ledger"]:
            if rec["source"] == "generated":
                lines.append(f"- {rec['id']}: \"{rec['copy']}\" "
                             f"({rec['targets_met']}/{rec['targets_total']} attributes confirmed)")
    path.write_text("\n".join(lines) + "\n")
