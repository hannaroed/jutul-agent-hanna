"""Tests for the unit-aware grader and the question/answer files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from grader import canonical_unit, convert, grade, parse_answer  # noqa: E402

QUANTITY = {"type": "quantity", "unit": "bar"}


def test_converts_pressure_and_grades_within_one_percent() -> None:
    expected = {"value": 167.32, "unit": "bar"}
    assert grade({"value": 16_732_000, "unit": "Pa"}, expected, QUANTITY).correct
    assert grade({"value": 168.9, "unit": "bar"}, expected, QUANTITY).correct
    assert not grade({"value": 169.1, "unit": "bar"}, expected, QUANTITY).correct


def test_rejects_wrong_dimension_and_missing_fields() -> None:
    expected = {"value": 167.32, "unit": "bar"}
    assert not grade({"value": 167.32, "unit": "kg"}, expected, QUANTITY).correct
    assert not grade({"choice": "x"}, expected, QUANTITY).correct
    assert not grade(None, expected, QUANTITY).correct


def test_percent_and_fraction_are_the_same_dimension() -> None:
    spec = {"type": "quantity", "unit": "%"}
    assert grade({"value": 0.1627, "unit": "fraction"}, {"value": 16.27, "unit": "%"}, spec).correct


def test_percentage_points_accept_a_percent_sign() -> None:
    spec = {"type": "quantity", "unit": "percentage_point"}
    expected = {"value": 22.27, "unit": "percentage_point"}
    assert grade({"value": 22.3, "unit": "%"}, expected, spec).correct
    assert grade({"value": 22.3, "unit": "pp"}, expected, spec).correct


def test_absolute_tolerance_for_counts() -> None:
    spec = {"type": "quantity", "unit": "1", "tolerance": {"abs": 0}}
    assert grade({"value": 8, "unit": ""}, {"value": 8.0, "unit": "1"}, spec).correct
    assert not grade({"value": 9, "unit": ""}, {"value": 8.0, "unit": "1"}, spec).correct
    # The count's noun as its unit (seen in the pilot).
    assert grade({"value": 8, "unit": "scenarios"}, {"value": 8.0, "unit": "1"}, spec).correct
    # ...but a real unit of another dimension is still wrong.
    assert not grade({"value": 8, "unit": "bar"}, {"value": 8.0, "unit": "1"}, spec).correct


def test_choices_ignore_case_spacing_and_order() -> None:
    spec = {"type": "choice"}
    assert grade({"choice": "prod 2"}, {"choice": "PROD2"}, spec).correct
    assert grade({"choice": "Scenario_07"}, {"choice": "scenario_07"}, spec).correct
    assert grade({"choice": "b, a"}, {"choice": "a,b"}, spec).correct
    assert not grade({"choice": "PROD1"}, {"choice": "PROD2"}, spec).correct


def test_unit_aliases() -> None:
    assert canonical_unit("tonnes") == "t"
    assert canonical_unit("m^3/day") == "m3/day"
    assert canonical_unit("Years") == "yr"
    assert convert(1.0, "t/day", "kg/day") == pytest.approx(1000.0)
    with pytest.raises(ValueError):
        canonical_unit("furlongs")


def test_parse_answer_takes_the_last_json_object() -> None:
    text = 'First {"value": 1, "unit": "bar"} then, corrected: {"value": 2, "unit": "bar"}.'
    assert parse_answer(text) == {"value": 2, "unit": "bar"}
    assert parse_answer("no json here") is None
    assert parse_answer('{"note": "x"} {"choice": "PROD2"}') == {"choice": "PROD2"}


def _questions() -> list[dict]:
    out = []
    for path in sorted((BENCH / "questions").glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        out += [dict(q, dataset=doc["dataset"]) for q in doc["questions"]]
    return out


def test_every_question_has_an_oracle_answer_in_a_known_unit() -> None:
    answers = json.loads((BENCH / "answers.json").read_text())
    questions = _questions()
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(answers), "questions and answers.json disagree"
    for q in questions:
        answer = answers[q["id"]]
        assert answer["dataset"] == q["dataset"], q["id"]
        assert q["tier"] in ("easy", "medium", "hard"), q["id"]
        if q["answer"]["type"] == "quantity":
            # The oracle's own value grades as correct against its question.
            assert convert(1.0, answer["unit"], q["answer"]["unit"]) > 0, q["id"]
            assert grade(answer, answer, q["answer"]).correct, q["id"]
        else:
            assert "choice" in answer, q["id"]
