"""Campaign artifacts and observed retrieval order."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

from jutul_agent.eval.trace_checks import finite_number

CSV_COLUMNS = (
    "run_id",
    "porosity",
    "injection_rate_m3_day",
    "recovery_factor",
    "avg_pressure_bar",
    "max_pressure_bar",
    "cell_id",
    "cell_pressure_bar",
    "water_saturation",
)
LEDGER_COLUMNS = (*CSV_COLUMNS[:6], "artifact_path")


def csv_rows(text: str, columns: tuple[str, ...]) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text), strict=True)
    if tuple(reader.fieldnames or ()) != columns:
        raise ValueError("CSV schema mismatch")
    rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError("ragged CSV row")
    return rows


def output_csv(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("@@ ") and line.endswith(" @@"):
            lines = lines[index + 1 :]
            break
    return "\n".join(re.sub(r"^\s*\d+(?::\s*|\t| {2})", "", line) for line in lines)


def artifact_rows(path: Path, case: str, cell_count: int = 192) -> list[dict[str, str]]:
    rows = csv_rows(path.read_text(encoding="utf-8"), CSV_COLUMNS)
    if len(rows) != cell_count or {row["cell_id"] for row in rows} != {
        str(cell) for cell in range(1, cell_count + 1)
    }:
        raise ValueError(f"{case}: cells missing, duplicated, or unexpected")
    for row in rows:
        if row["run_id"] != case:
            raise ValueError(f"{case}: wrong run ID")
        numbers = {key: float(row[key]) for key in CSV_COLUMNS[1:]}
        if not all(finite_number(value) for value in numbers.values()):
            raise ValueError(f"{case}: non-finite value")
        if not 0 < numbers["porosity"] < 1 or numbers["injection_rate_m3_day"] <= 0:
            raise ValueError(f"{case}: invalid parameters")
        if any(
            numbers[key] <= 0
            for key in ("cell_pressure_bar", "avg_pressure_bar", "max_pressure_bar")
        ):
            raise ValueError(f"{case}: non-positive absolute pressure")
        if not 0 <= numbers["water_saturation"] <= 1 or not 0 <= numbers["recovery_factor"] <= 1:
            raise ValueError(f"{case}: invalid saturation/recovery")
        if any(row[key] != rows[0][key] for key in CSV_COLUMNS[:6]):
            raise ValueError(f"{case}: inconsistent run summary")
    pressures = [float(row["cell_pressure_bar"]) for row in rows]
    if abs(max(pressures) - float(rows[0]["max_pressure_bar"])) > 1e-6:
        raise ValueError(f"{case}: maximum pressure disagrees with cells")
    if abs(sum(pressures) / len(pressures) - float(rows[0]["avg_pressure_bar"])) > 1e-6:
        raise ValueError(f"{case}: average pressure disagrees with cells")
    return rows


def compare_reference(rows: list[dict], reference: list[dict]) -> None:
    if len(rows) != len(reference):
        raise ValueError("reference row count mismatch")
    expected = {row["cell_id"]: row for row in reference}
    for row in rows:
        truth = expected.get(row["cell_id"])
        if truth is None or row["run_id"] != truth["run_id"]:
            raise ValueError("reference row identity mismatch")
        for key in CSV_COLUMNS[1:]:
            tolerance = 1e-4 if "pressure" in key else 1e-7
            actual, target = float(row[key]), float(truth[key])
            if not finite_number(actual) or not finite_number(target):
                raise ValueError(f"{row['run_id']}: non-finite reference comparison")
            if abs(actual - target) > tolerance:
                raise ValueError(f"{row['run_id']}: {key} differs from calibrated reference")


def validate_ledger(rows: list[dict], artifacts: dict[str, list[dict]]) -> None:
    if len(rows) != len(artifacts) or {row["run_id"] for row in rows} != set(artifacts):
        raise ValueError("ledger completed set mismatch")
    for row in rows:
        case = row["run_id"]
        if row["artifact_path"] != f"results/{case}.csv":
            raise ValueError("ledger artifact locator mismatch")
        if any(row[key] != artifacts[case][0][key] for key in CSV_COLUMNS[:6]):
            raise ValueError("ledger values disagree with artifact")
