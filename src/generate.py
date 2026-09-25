"""Write new copy from the winning attributes, then check it with Jev.

Winning attributes are the significant ones from the stats step. Claude writes
new copy that uses all of them, Jev tags the new copy to confirm it actually
carries those attributes, and everything lands in a ledger with IDs so new copy
can later be scored against how it performs.
"""

import json
import re

from discover import call_claude
from tag import tag_rows


def winners(results, config):
    """Significant group comparisons, with the winning option and its meaning."""
    by_name = {a["name"]: a for a in config["attributes"]}
    out = []
    for r in results:
        if r["significant"] != "yes" or not r["groups"] or r["attribute"] not in by_name:
            continue
        a = by_name[r["attribute"]]
        best = max(r["groups"], key=lambda g: r["groups"][g]["mean"])
        if a["type"] == "choice":
            meaning = a["options"].get(best, best)
        else:
            meaning = a["true"] if best == "yes" else a["false"]
        out.append({"name": a["name"], "value": best, "meaning": meaning,
                    "effect_pts": round(r["effect_pts"], 2)})
    return out


def write_copy(config, wins, examples, metric, api_key):
    g = config["generation"]
    prompt = g["prompt"].format(
        count=g["count"], metric=metric.replace("_", " "),
        winners="\n".join(f"- {w['name']} = {w['value']}: {w['meaning']}" for w in wins),
        examples="\n".join(f"- {e}" for e in examples[:8]))
    reply, _, _ = call_claude(prompt, g["model"], api_key)
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", reply.strip())
    return prompt, [c.strip() for c in json.loads(body)][: g["count"]]


def build_ledger(existing, new_copy, checked, wins):
    """One record per piece of copy: IDs for existing and new, source, attributes."""
    names = [w["name"] for w in wins]
    ledger = []
    for i, r in enumerate(existing, 1):
        ledger.append({"id": f"existing_{i:03d}", "source": "existing", "copy": r["copy"],
                       "conversion_rate": float(r["conversion_rate"]),
                       "attributes": {n: r.get(f"jev_{n}") for n in names}})
    for i, (text, t) in enumerate(zip(new_copy, checked), 1):
        attrs = {n: t.get(f"jev_{n}") for n in names}
        ledger.append({
            "id": f"new_{i:03d}", "source": "generated", "copy": text,
            "conversion_rate": None,  # filled in once it has run
            "attributes": attrs,
            "targets_met": sum(attrs[w["name"]] == w["value"] for w in wins),
            "targets_total": len(wins)})
    return ledger


def generate(config, results, tagged_rows, metric, api_key):
    wins = winners(results, config)
    if not wins:
        return {"winners": [], "new_copy": [], "ledger": []}
    top = sorted(tagged_rows, key=lambda r: -float(r[metric]))
    prompt, new_copy = write_copy(config, wins, [r["copy"] for r in top], metric, api_key)
    check_config = dict(config)
    check_config["attributes"] = [a for a in config["attributes"]
                                  if a["name"] in {w["name"] for w in wins}]
    checked = tag_rows([{"copy": c} for c in new_copy], check_config, api_key)
    return {"winners": wins, "prompt": prompt, "new_copy": new_copy,
            "ledger": build_ledger(tagged_rows, new_copy, checked, wins)}
