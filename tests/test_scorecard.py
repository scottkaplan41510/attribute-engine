"""The scorecard must find what's really there and nothing else.

labeled_sample.csv is a real Jev-labeled run of the 30 sample ads. The sample
was generated with exactly two real patterns (solution framing, a specific
number); everything else is noise. No API calls: these run offline in seconds.

Run: pip install pytest && pytest
"""

import csv
import random
from pathlib import Path

from attribute_engine.stats import scorecard

JEV = ["framing", "cta_type", "second_sentence_focus", "third_sentence_proof_type",
       "uses_specific_number", "opening_is_question"]


def cfg(*jev, code=()):
    return {"attributes": [{"name": n, "computed_by": "code"} for n in code]
            + [{"name": n, "computed_by": "jev"} for n in jev]}


def sample():
    return list(csv.DictReader(open(Path(__file__).parent / "labeled_sample.csv")))


def significant(card):
    return {(r["attribute"], r["value"]) for r in card if r["significant"]}


def test_finds_the_two_planted_patterns_and_nothing_else():
    card = scorecard(sample(), cfg(*JEV, code=["word_count"]))
    assert significant(card) == {("framing", "solution"), ("framing", "problem"),
                                 ("uses_specific_number", "yes"),
                                 ("uses_specific_number", "no")}


def test_numbers_match_the_published_example():
    card = {(r["attribute"], r["value"]): r for r in scorecard(sample(), cfg(*JEV))}
    solution = card[("framing", "solution")]
    assert (solution["n_with"], solution["n_rest"]) == (15, 15)
    assert round(100 * solution["mean_with"], 1) == 8.5
    assert round(100 * solution["mean_rest"], 1) == 4.6
    assert round(solution["correlation"], 2) == 0.72


def test_picking_the_best_cta_does_not_make_noise_significant():
    # Book demo is the best of three CTAs by chance. Tested alone against the
    # rest it looks significant; the all-values gate must stop that.
    card = {(r["attribute"], r["value"]): r for r in scorecard(sample(), cfg(*JEV))}
    assert card[("cta_type", "book demo")]["correlation"] > 0.4
    assert not card[("cta_type", "book demo")]["significant"]


def test_two_strong_values_can_both_be_significant():
    random.seed(1)
    means = {"tech": 0.09, "healthcare": 0.087, "retail": 0.06,
             "entertainment": 0.04, "finance": 0.06}
    rows = [{"conversion_rate": m + random.gauss(0, 0.008), "jev_industry": k}
            for k, m in means.items() for _ in range(10)]
    sig = significant(scorecard(rows, cfg("industry")))
    assert {("industry", "tech"), ("industry", "healthcare")} <= sig
    assert ("industry", "retail") not in sig


def test_an_attribute_with_one_value_still_shows():
    rows = [{**r, "jev_sentiment": "positive"} for r in sample()]
    rows_out = [r for r in scorecard(rows, cfg("sentiment")) if r["attribute"] == "sentiment"]
    assert len(rows_out) == 1
    assert rows_out[0]["p_adjusted"] is None and not rows_out[0]["significant"]
    assert "nothing to compare" in rows_out[0]["note"]


def test_one_row_in_a_group_shows_with_no_p_value():
    rows = sample()
    for i, r in enumerate(rows):
        r["jev_sentiment"] = "negative" if i == 0 else "positive"
    out = [r for r in scorecard(rows, cfg("sentiment")) if r["attribute"] == "sentiment"]
    assert {r["value"] for r in out} == {"positive", "negative"}
    assert all(r["p_adjusted"] is None for r in out)
    assert all("Too few rows" in r["note"] for r in out)
