"""Write new copy from the winning attributes, then check it with Jev.

Winning attributes are the significant ones from the stats step. Claude writes
new copy that uses all of them, plus a control set that deliberately uses the
losing values. Jev tags both to confirm they carry what they should, and every
piece of copy lands in a creative log with an ID, so performance can be filled
in later and the approach validated (see validate.py).
"""

import csv
import json
import re

from .discover import call_claude
from .tag import tag_rows


def winners(results, config, pick=max):
    """Significant group comparisons, with the winning option and its meaning.
    pick=min returns the losing option instead, for the control set."""
    by_name = {a["name"]: a for a in config["attributes"]}
    out = []
    for r in results:
        if r["significant"] != "yes" or not r["groups"] or r["attribute"] not in by_name:
            continue
        a = by_name[r["attribute"]]
        best = pick(r["groups"], key=lambda g: r["groups"][g]["mean"])
        if a["type"] == "choice":
            meaning = a["options"].get(best, best)
        else:
            meaning = a["true"] if best == "yes" else a["false"]
        out.append({"name": a["name"], "value": best, "meaning": meaning,
                    "effect_pts": round(r["effect_pts"], 2)})
    return out


def write_copy(config, attrs, examples, metric, api_key, prompt_key="prompt", count_key="count"):
    g = config["generation"]
    count = g[count_key]
    prompt = g[prompt_key].format(
        count=count, metric=metric.replace("_", " "),
        winners="\n".join(f"- {w['name']} = {w['value']}: {w['meaning']}" for w in attrs),
        examples="\n".join(f"- {e}" for e in examples[:8]))
    reply, _, _ = call_claude(prompt, g["model"], api_key)
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())
    return prompt, [c.strip() for c in json.loads(body)][:count]


def build_ledger(existing, made, wins, metric):
    """One record per piece of copy: ID, source, copy, attributes, metric.

    made is a list of (source, copy, jev_row) for generated and control copy.
    Their metric is empty until the copy has run and someone fills it in.
    """
    names = [w["name"] for w in wins]
    ledger = []
    for i, r in enumerate(existing, 1):
        ledger.append({"id": f"original_{i:03d}", "source": "original", "copy": r["copy"],
                       metric: float(r[metric]),
                       "attributes": {n: r.get(f"jev_{n}") for n in names}})
    counters = {}
    for source, text, t in made:
        counters[source] = counters.get(source, 0) + 1
        attrs = {n: t.get(f"jev_{n}") for n in names}
        ledger.append({
            "id": f"{source}_{counters[source]:03d}", "source": source, "copy": text,
            metric: None,
            "attributes": attrs,
            "matches_winners": sum(attrs[w["name"]] == w["value"] for w in wins),
            "winners_total": len(wins)})
    return ledger


def write_log(ledger, metric, path):
    """The creative log: one row per piece of copy, metric column to fill in."""
    names = sorted({n for rec in ledger for n in rec["attributes"]})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "copy", *names, metric, "notes"])
        for rec in ledger:
            value = rec[metric]
            w.writerow([rec["id"], rec["source"], rec["copy"],
                        *[rec["attributes"].get(n) for n in names],
                        "" if value is None else value, ""])


def generate(config, results, tagged_rows, metric, api_key):
    wins = winners(results, config)
    if not wins:
        return {"winners": [], "new_copy": [], "controls": [], "ledger": []}
    losers = winners(results, config, pick=min)
    examples = [r["copy"] for r in sorted(tagged_rows, key=lambda r: -float(r[metric]))]
    prompt, new_copy = write_copy(config, wins, examples, metric, api_key)
    _, controls = write_copy(config, losers, examples, metric, api_key,
                             prompt_key="control_prompt", count_key="control_count")
    check_config = dict(config)
    check_config["attributes"] = [a for a in config["attributes"]
                                  if a["name"] in {w["name"] for w in wins}]
    checked = tag_rows([{"copy": c} for c in new_copy + controls], check_config, api_key)
    made = ([("generated", c, t) for c, t in zip(new_copy, checked[:len(new_copy)])]
            + [("control", c, t) for c, t in zip(controls, checked[len(new_copy):])])
    return {"winners": wins, "losers": losers, "prompt": prompt, "new_copy": new_copy,
            "controls": controls, "ledger": build_ledger(tagged_rows, made, wins, metric)}
