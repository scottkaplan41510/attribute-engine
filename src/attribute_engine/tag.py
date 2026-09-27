"""Tag every row with its attributes.

Code attributes (counts, lengths) are measured in Python. Jev attributes are
answered by Jev through Vercel AI Gateway: one call per row, every Jev
question in that single call. Jev returns a weight for every option, and we
keep all of them. Nothing is dropped for low confidence; it is only flagged.

Usage: python -m attribute_engine.tag --input data/sample_ads.csv --output output/tags.csv
"""

import argparse
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

ROOT = Path.cwd()  # data/, output/ and .env are read from where you run it
CONFIG = Path(__file__).resolve().parent / "attributes.yaml"
GATEWAY_URL = "https://ai-gateway.vercel.sh/v1/evaluate"
OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"
MAX_ATTEMPTS = 4  # with 1s, 2s, 4s waits: gives up after ~7s of retrying


def load_config(path=CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def jev_attributes(config):
    return [a for a in config["attributes"] if a["computed_by"] == "jev"]


def build_questions(attrs, provider="vercel"):
    """Translate attributes.yaml into Jev's question schema."""
    yes_no = "noul" if provider == "openrouter" else "boolean"
    questions = {}
    for a in attrs:
        q = {"type": yes_no if a["type"] == "boolean" else a["type"],
             "instructions": a["question"]}
        if a["type"] == "choice":
            q["criteria"] = dict(a["options"])
        elif a["type"] == "boolean":
            q["criteria"] = {"true": a["true"], "false": a["false"]}
        questions[a["name"]] = q
    return questions


def build_request(text, questions, model, provider_order=None):
    request = {"model": model, "state": text, "questions": questions}
    if provider_order and not model.startswith("typesafe/"):
        request["providerOptions"] = {"gateway": {"order": list(provider_order)}}
    return request


def code_attributes(text):
    words = re.findall(r"[A-Za-z0-9$%.,'-]+", text)
    words = [w.strip(".,") for w in words if w.strip(".,")]
    return {"word_count": len(words)}


class JevBusyError(RuntimeError):
    """Jev (or the gateway in front of it) kept saying it was overloaded."""


def call_jev(payload, api_key, url=GATEWAY_URL):
    """One Jev call. Retries transient failures, then raises."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        r = None
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=20)
            seconds = time.perf_counter() - started
            if r.status_code == 200:
                return r.json(), seconds
            if r.status_code in (429, 503) and attempt == MAX_ATTEMPTS:
                raise JevBusyError(f"Jev is busy ({r.status_code}): {r.text[:200]}")
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"Jev call failed ({r.status_code}): {r.text[:300]}")
        except requests.RequestException:
            if attempt == MAX_ATTEMPTS:
                raise
        retry_after = r.headers.get("retry-after") if r is not None else None
        time.sleep(min(float(retry_after) if retry_after else 2 ** (attempt - 1), 4))


def parse_answers(response, attrs, low_confidence):
    """Flatten Jev's answers into columns: label, weights, confidence, flag."""
    out = {}
    for a in attrs:
        ans = response["answers"][a["name"]]
        name = a["name"]
        if a["type"] == "choice":
            weights = ans["probabilities"]
            out[f"jev_{name}"] = ans["choice"]
            out[f"jev_{name}_weights"] = json.dumps(weights)
            conf = ans.get("confidence", max(weights.values()))
        else:  # boolean: probability that the answer is yes
            p = ans.get("probability", ans.get("noul"))
            out[f"jev_{name}"] = "yes" if p >= 0.5 else "no"
            out[f"jev_{name}_p"] = round(p, 4)
            conf = max(p, 1 - p)  # Jev returns no separate confidence for booleans
        out[f"jev_{name}_confidence"] = round(conf, 4)
        out[f"jev_{name}_low_confidence"] = "yes" if conf < low_confidence else "no"
    return out


def jev_route(settings, api_key):
    """Where Jev calls go: OpenRouter or Vercel AI Gateway (attributes.yaml)."""
    if settings.get("jev_provider") == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY in .env (see .env.example).")
        return "openrouter", OPENROUTER_URL, settings["openrouter_model"], key
    return "vercel", GATEWAY_URL, settings["model"], api_key


def tag_rows(rows, config, api_key):
    attrs = jev_attributes(config)
    settings = config["settings"]
    provider, url, model, key = jev_route(settings, api_key)
    questions = build_questions(attrs, provider)

    def work(row):
        response, seconds = call_jev(
            build_request(row["copy"], questions, model, settings.get("provider_order")),
            key, url)
        cost = response.get("providerMetadata", {}).get("gateway", {})
        usage = response.get("usage", {})
        return {
            **row,
            **code_attributes(row["copy"]),
            **parse_answers(response, attrs, settings["low_confidence"]),
            "jev_input_tokens": usage.get("inputTokens", usage.get("input_tokens", 0)),
            "jev_output_tokens": usage.get("outputTokens", usage.get("output_tokens", 0)),
            "jev_market_cost_usd": float(cost.get("marketCost", usage.get("cost", 0)) or 0),
            "jev_billed_cost_usd": float(cost.get("cost", usage.get("cost", 0)) or 0),
            "jev_seconds": round(seconds, 3),
        }

    done = []

    def work_with_progress(row):
        result = work(row)
        done.append(1)
        if len(done) % 25 == 0:
            print(f"  {len(done)}/{len(rows)} tagged", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=settings["concurrency"]) as pool:
        return list(pool.map(work_with_progress, rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "data/sample_ads.csv"))
    parser.add_argument("--output", default=str(ROOT / "output/tags.csv"))
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    api_key = os.environ["AI_GATEWAY_API_KEY"]
    config = load_config(args.config)
    rows = list(csv.DictReader(open(args.input)))

    started = time.perf_counter()
    tagged = tag_rows(rows, config, api_key)
    wall = time.perf_counter() - started

    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tagged[0]))
        w.writeheader()
        w.writerows(tagged)

    market = sum(t["jev_market_cost_usd"] for t in tagged)
    billed = sum(t["jev_billed_cost_usd"] for t in tagged)
    tokens = sum(t["jev_input_tokens"] for t in tagged)
    print(f"Tagged {len(tagged)} rows in {wall:.1f}s "
          f"({tokens:,} input tokens, ${market:.4f} at list price, ${billed:.4f} billed)")
    for a in jev_attributes(config):
        low = sum(t[f"jev_{a['name']}_low_confidence"] == "yes" for t in tagged)
        print(f"  {a['name']:16} low-confidence answers: {low}")


if __name__ == "__main__":
    main()
