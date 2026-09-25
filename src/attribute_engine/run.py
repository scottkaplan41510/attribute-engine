"""Run the whole pipeline with one command.

  1. Tag every row (code attributes + Jev)
  2. Stats, round 1
  3. Blind discovery: Claude proposes new attributes from the copy alone
  4. Jev tags the approved attributes, stats round 2
  5. Claude writes new copy from the winning attributes; Jev checks it

Usage: python -m attribute_engine.run --input data/sample_ads.csv [--auto-approve]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .discover import discover
from .generate import generate
from .report import write_report
from .stats import METRIC, run as run_stats, write as write_results
from .tag import tag_rows

ROOT = Path.cwd()  # data/, output/ and .env are read from where you run it
CONFIG = Path(__file__).resolve().parent / "attributes.yaml"
OUT = ROOT / "output"


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def step(n, text):
    print(f"\n[{n}/5] {text}")


def analyze(rows, key, config, auto_approve=True, write_copy=True, log=lambda n, t: None):
    """The whole pipeline on a list of {copy, conversion_rate} rows. Used by the
    command line and by the hosted web tool, so both run the same logic."""
    log(1, f"Tagging {len(rows)} rows")
    tagged = tag_rows(rows, config, key)
    log(2, "Stats, round 1")
    round1 = run_stats(tagged, config)
    log(3, "Blind discovery (Claude sees the copy only, never the metric)")
    found = discover(config, [r["copy"] for r in rows], auto_approve)
    config2 = dict(config)
    config2["attributes"] = config["attributes"] + found["approved"]
    log(4, f"Tagging {len(found['approved'])} discovered attributes, stats round 2")
    tagged2 = tag_rows(rows, config2, key)
    round2 = run_stats(tagged2, config2)
    made = {"winners": [], "new_copy": [], "ledger": []}
    if write_copy:
        log(5, "Writing new copy from the winning attributes")
        made = generate(config2, round2, tagged2, METRIC, key)
    return {"tagged": tagged, "round1": round1, "discovery": found,
            "config_round2": config2, "tagged_round2": tagged2, "round2": round2,
            "generated": made}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "data/sample_ads.csv"))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--auto-approve", action="store_true",
                        help="keep every discovered attribute without asking")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.environ.get("AI_GATEWAY_API_KEY")
    if not key:
        sys.exit("Set AI_GATEWAY_API_KEY in .env (see .env.example).")
    config = yaml.safe_load(open(args.config))
    rows = list(csv.DictReader(open(args.input)))
    for col in ("copy", METRIC):
        if col not in rows[0]:
            sys.exit(f"Input needs a '{col}' column.")
    OUT.mkdir(exist_ok=True)

    result = analyze(rows, key, config, auto_approve=args.auto_approve, log=step)
    save_csv(result["tagged"], OUT / "tags.csv")
    write_results(result["round1"], OUT / "results.csv")
    (OUT / "discovery_prompt.txt").write_text(result["discovery"]["prompt"])
    (OUT / "discovery_reply.txt").write_text(result["discovery"]["reply"])
    yaml.safe_dump(result["config_round2"], open(OUT / "attributes_round2.yaml", "w"),
                   sort_keys=False)
    save_csv(result["tagged_round2"], OUT / "tags_round2.csv")
    write_results(result["round2"], OUT / "results_round2.csv")
    json.dump(result["generated"]["ledger"], open(OUT / "ledger.json", "w"), indent=2)

    found, made, round1, round2 = (result["discovery"], result["generated"],
                                   result["round1"], result["round2"])
    write_report(round1, round2, found, made, len(rows), OUT / "report.md")
    print(f"\nDone. See {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
