"""Blind discovery: Claude proposes new attributes from the copy alone.

Claude never sees the performance metric. It returns up to N attributes in the
attributes.yaml format. The user approves which to keep (or --auto-approve),
and the approved ones are written to a round-2 config for Jev to tag.

Usage: python src/discover.py --input data/sample_ads.csv [--auto-approve]
"""

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CHAT_URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def build_prompt(config, texts):
    d = config["discovery"]
    existing = "\n".join(
        f"- {a['name']}: " + (a.get("question") or "measured in code")
        for a in config["attributes"])
    return d["prompt"].format(
        n=len(texts), existing=existing, max_attributes=d["max_attributes"],
        copy="\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts)))


def call_claude(prompt, model, api_key):
    """Claude directly when ANTHROPIC_API_KEY is set, else via Vercel AI Gateway.
    Returns (reply_text, input_tokens, output_tokens)."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        r = requests.post(ANTHROPIC_URL, timeout=120, headers={
            "x-api-key": anthropic_key, "anthropic-version": "2023-06-01"},
            json={"model": model.split("/")[-1], "max_tokens": 8000,
                  "messages": [{"role": "user", "content": prompt}]})
        if r.status_code != 200:
            raise RuntimeError(f"Claude call failed ({r.status_code}): {r.text[:300]}")
        body = r.json()
        text = "".join(b["text"] for b in body["content"] if b.get("type") == "text")
        return (text, body["usage"]["input_tokens"],
                body["usage"]["output_tokens"])
    r = requests.post(CHAT_URL, timeout=120,
                      headers={"Authorization": f"Bearer {api_key}"},
                      json={"model": model, "max_tokens": 8000,
                            "messages": [{"role": "user", "content": prompt}]})
    if r.status_code != 200:
        raise RuntimeError(f"Claude call failed ({r.status_code}): {r.text[:300]}")
    body = r.json()
    return (body["choices"][0]["message"]["content"], body["usage"]["prompt_tokens"],
            body["usage"]["completion_tokens"])


def parse_attributes(text, max_attributes):
    body = re.sub(r"^```(?:yaml)?\s*|\s*```$", "", text.strip())
    proposed = yaml.safe_load(body) or []
    out = []
    for a in proposed[:max_attributes]:
        a["computed_by"] = "jev"
        if a["type"] == "boolean":  # YAML may read true/false keys as booleans
            a["true"] = a.pop(True, a.get("true"))
            a["false"] = a.pop(False, a.get("false"))
        out.append(a)
    return out


def discover(config, texts, auto_approve=False):
    """Ask Claude for new attributes from the copy alone; return what was approved."""
    d = config["discovery"]
    texts = random.Random(0).sample(texts, min(len(texts), d["sample_size"]))
    prompt = build_prompt(config, texts)
    reply, tokens_in, tokens_out = call_claude(
        prompt, d["model"], os.environ.get("AI_GATEWAY_API_KEY"))
    proposed = parse_attributes(reply, d["max_attributes"])
    print(f"Claude proposed {len(proposed)} attributes "
          f"({tokens_in} in / {tokens_out} out tokens):\n")
    approved = []
    for i, a in enumerate(proposed, 1):
        opts = " | ".join(a["options"]) if a["type"] == "choice" else "yes | no"
        print(f"  {i}. {a['name']} ({opts})\n     {a['question']}\n     why: {a.get('why', '')}")
        if auto_approve or input("     keep? [y/n] ").strip().lower() == "y":
            approved.append({k: v for k, v in a.items() if k != "why"})
    return {"prompt": prompt, "reply": reply, "proposed": proposed, "approved": approved,
            "tokens_in": tokens_in, "tokens_out": tokens_out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "data/sample_ads.csv"))
    parser.add_argument("--auto-approve", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = yaml.safe_load(open(ROOT / "attributes.yaml"))
    texts = [r["copy"] for r in csv.DictReader(open(args.input))]  # copy only
    result = discover(config, texts, args.auto_approve)

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "discovery_prompt.txt").write_text(result["prompt"])
    (out_dir / "discovery_reply.txt").write_text(result["reply"])
    round2 = dict(config)
    round2["attributes"] = config["attributes"] + result["approved"]
    yaml.safe_dump(round2, open(out_dir / "attributes_round2.yaml", "w"), sort_keys=False)
    print(f"\nApproved {len(result['approved'])}. Round-2 config: output/attributes_round2.yaml")


if __name__ == "__main__":
    main()
