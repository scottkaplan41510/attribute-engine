"""Part 3: the same labeling job, done by Jev and by Claude models.

Every model labels the same rows with the same questions. We measure what it
cost, how long it took, and how often it matched the answer key.

Usage: attribute-engine-compare [--input data/true_labels.csv]
"""

import argparse
import copy
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

from .tag import CONFIG, tag_rows

ROOT = Path.cwd()
CHAT_URL = "https://ai-gateway.vercel.sh/v1/chat/completions"


def claude_prompt(text, attrs):
    lines = [f"Label this marketing copy.\n\nCopy: {text}\n\nAnswer each question:"]
    for a in attrs:
        if a["type"] == "choice":
            opts = "; ".join(f"{k} = {v}" for k, v in a["options"].items())
            lines.append(f'- "{a["name"]}": {a["question"]} One of: {opts}')
        else:
            lines.append(f'- "{a["name"]}": {a["question"]} Answer "yes" or "no".')
    keys = ", ".join(f'"{a["name"]}": "..."' for a in attrs)
    lines.append(f"\nReturn only JSON like {{{keys}}}.")
    return "\n".join(lines)


def label_with_claude(rows, attrs, model, key, concurrency):
    def one(row):
        r = requests.post(CHAT_URL, timeout=120, headers={"Authorization": f"Bearer {key}"},
                          json={"model": model, "max_tokens": 2000,
                                "messages": [{"role": "user",
                                              "content": claude_prompt(row["copy"], attrs)}]})
        r.raise_for_status()
        body = r.json()
        text = body["choices"][0]["message"]["content"]
        answer = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        usage = body.get("usage", {})
        return answer, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    with ThreadPoolExecutor(concurrency) as pool:
        return list(pool.map(one, rows))


def accuracy(predicted, truth, attrs):
    scores = {}
    for a in attrs:
        n = a["name"]
        scores[n] = sum(str(p.get(n, "")).lower() == t[n] for p, t in zip(predicted, truth)) / len(truth)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "data/true_labels.csv"))
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    gateway_key = os.environ["AI_GATEWAY_API_KEY"]

    config = yaml.safe_load(open(CONFIG))
    cmp = config["compare"]
    attrs = cmp["attributes"]
    truth = list(csv.DictReader(open(args.input)))
    rows = [{"copy": t["copy"]} for t in truth]
    conc = config["settings"]["concurrency"]
    results = []
    per_row = [{"copy": t["copy"], **{f"truth_{a['name']}": t[a["name"]] for a in attrs}} for t in truth]

    # Jev
    jev_config = copy.deepcopy(config)
    jev_config["attributes"] = attrs
    start = time.perf_counter()
    tagged = tag_rows(rows, jev_config, gateway_key)
    seconds = time.perf_counter() - start
    predicted = [{a["name"]: t[f"jev_{a['name']}"] for a in attrs} for t in tagged]
    for row, p in zip(per_row, predicted):
        row.update({f"jev_{k}": v for k, v in p.items()})
    results.append({"model": "Jev (TypeSafe)", "seconds": seconds,
                    "cost_usd": sum(t["jev_billed_cost_usd"] for t in tagged),
                    **accuracy(predicted, truth, attrs)})

    # Claude models
    for model in cmp["claude_models"]:
        start = time.perf_counter()
        out = label_with_claude(rows, attrs, model, gateway_key, conc)
        seconds = time.perf_counter() - start
        price_in, price_out = cmp["prices_per_million"][model]
        cost = sum(i * price_in + o * price_out for _, i, o in out) / 1_000_000
        short = model.split("/")[-1]
        for row, (p, _, _) in zip(per_row, out):
            row.update({f"{short}_{k}": v for k, v in p.items()})
        results.append({"model": short, "seconds": seconds, "cost_usd": cost,
                        **accuracy([a for a, _, _ in out], truth, attrs)})

    out_path = ROOT / "output/model_comparison.csv"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    with open(ROOT / "output/model_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_row[0]))
        w.writeheader()
        w.writerows(per_row)

    names = [a["name"] for a in attrs]
    print(f"Labeling {len(rows)} rows, {len(attrs)} questions each:\n")
    print(f"  {'model':22} {'cost':>10} {'time':>8}  " + "  ".join(f"{n:>16}" for n in names))
    for r in results:
        print(f"  {r['model']:22} ${r['cost_usd']:>9.5f} {r['seconds']:>7.1f}s  "
              + "  ".join(f"{100 * r[n]:>15.0f}%" for n in names))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
