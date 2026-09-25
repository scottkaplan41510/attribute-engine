"""Build synthetic B2B ads with planted patterns, so the right answer is known
before the engine runs.

Each ad is four sentences: opener (sets framing), middle (sets value focus and
technical level), proof (a specific number or a vague stand-in of the same
length), and a call to action. Conversion rate is set by rule:

    baseline + FRAMING_EFFECT if solution-framed
             + NUMBER_EFFECT  if the proof mentions a specific number
             + random noise

Everything else (value focus, technical level, word count, word length, CTA)
has no effect and must come out not significant.

Writes:
    data/sample_ads.csv   copy, conversion_rate  (the engine's input)
    data/true_labels.csv  every ad with its true attribute labels
    data/hand_labels.csv  50 of those ads, for scoring tagging accuracy
"""

import csv
import random
from pathlib import Path

SEED = 41
N_ADS = 30
N_HAND_LABELS = 30

BASELINE = 0.030
FRAMING_EFFECT = 0.040  # +4.0 points for solution framing
NUMBER_EFFECT = 0.040   # +4.0 points for mentioning a specific number (hidden)
NOISE_SD = 0.008        # 0.8 points of noise per ad

# Framing mix: half problem-framed, half solution-framed.
FRAMING_WEIGHTS = {"solution": 0.5, "problem": 0.5}
NUMBER_RATE = 0.5

# Each sentence carries exactly one attribute and stays neutral on the others,
# so the answer key matches what a reader of the whole ad would say.
OPENERS = {
    "problem": [
        "Tired of wasted ad spend?",
        "Your ad budget is hard to control.",
        "Most B2B campaigns overspend quietly.",
    ],
    "solution": [
        "Take control of your ad budget.",
        "Fix overspending on every campaign.",
        "Run every campaign on budget.",
    ],
    "neither": [
        "Meet the new ad platform.",
        "Built for B2B marketing teams.",
        "A different way to run ads.",
    ],
}

# (value_focus, technical_level) -> middle sentences. No "neither": in a
# spot check Jev read any ad without clear features as benefit-led, and real
# ad copy almost always carries one or the other.
MIDDLES = {
    ("feature", "technical"): [
        "Includes API bid scripts and CPA pacing rules.",
        "Includes real-time pacing alerts via the Google Ads API.",
    ],
    ("feature", "plain"): [
        "Includes simple alerts and one shared dashboard.",
        "Includes a weekly report and a budget calendar.",
    ],
    ("benefit", "technical"): [
        "Your team hits CPA and ROAS targets with less effort.",
        "Your team sees steadier CPA and higher ROAS.",
    ],
    ("benefit", "plain"): [
        "Your team saves hours and sleeps better.",
        "Your team feels confident and stays focused.",
    ],
}

# (with a specific number, vague stand-in) of equal word count, so mentioning
# a number never changes word count.
PROOFS = [
    ("Trusted by 2,400 teams.", "Trusted by growing teams."),
    ("Setup takes 5 minutes.", "Setup takes mere minutes."),
    ("Rated 4.8 by marketers.", "Rated highly by marketers."),
    ("Over 90 integrations built in.", "Dozens of integrations built in."),
]

CTAS = ["Start your free trial.", "Book a demo today.", "Learn more."]

ROOT = Path.cwd()  # data/, output/ and .env are read from where you run it
DATA = ROOT / "data"


def _weighted(rng, weights):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def build_ads(seed=SEED, n=N_ADS):
    rng = random.Random(seed)
    ads, seen = [], set()
    while len(ads) < n:
        framing = _weighted(rng, FRAMING_WEIGHTS)
        value_focus = rng.choice(["feature", "benefit"])
        technical = rng.choice(["technical", "plain"])
        has_number = rng.random() < NUMBER_RATE

        with_num, vague = rng.choice(PROOFS)
        copy = " ".join([
            rng.choice(OPENERS[framing]),
            rng.choice(MIDDLES[(value_focus, technical)]),
            with_num if has_number else vague,
            rng.choice(CTAS),
        ])
        if copy in seen:
            continue
        seen.add(copy)

        rate = (BASELINE
                + (FRAMING_EFFECT if framing == "solution" else 0)
                + (NUMBER_EFFECT if has_number else 0)
                + rng.gauss(0, NOISE_SD))
        ads.append({
            "ad_id": len(ads) + 1,
            "copy": copy,
            "conversion_rate": round(max(rate, 0.001), 4),
            "framing": framing,
            "value_focus": value_focus,
            "technical_level": technical,
            "mentions_number": "yes" if has_number else "no",
        })

    hand = rng.sample(ads, N_HAND_LABELS)
    return ads, sorted(hand, key=lambda a: a["ad_id"])


def _write(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ads, hand = build_ads()
    DATA.mkdir(exist_ok=True)
    label_fields = ["ad_id", "copy", "conversion_rate", "framing", "value_focus",
                    "technical_level", "mentions_number"]
    _write(DATA / "sample_ads.csv", ads, ["copy", "conversion_rate"])
    _write(DATA / "true_labels.csv", ads, label_fields)
    _write(DATA / "hand_labels.csv", hand, label_fields)

    print(f"Wrote {len(ads)} ads and {len(hand)} hand labels to data/\n")
    for attr in ["framing", "value_focus", "technical_level", "mentions_number"]:
        groups = {}
        for a in ads:
            groups.setdefault(a[attr], []).append(a["conversion_rate"])
        summary = ", ".join(
            f"{k} {100 * sum(v) / len(v):.2f}% (n={len(v)})"
            for k, v in sorted(groups.items()))
        print(f"  {attr:16} {summary}")


if __name__ == "__main__":
    main()
