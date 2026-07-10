#!/usr/bin/env python3
"""
Collect ActiveFly evaluation outputs into eval/.

The collector is intentionally adapter-based:
- API JSONL files under ACTIVEFLY_RESULTS_DIR are one PredictionStore;
- future one-trajectory-at-a-time records are another PredictionStore.

Output behavior stays stable: missing files, missing choices, and missing metric
values are written as blank strings.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from activefly_eval_data import DATASETS
from activefly_eval_data import MODELS
from activefly_eval_data import ROOT
from activefly_eval_data import VALID
from activefly_eval_data import VIEWS
from activefly_eval_data import DatasetProvider
from activefly_eval_data import collect_fluc_rows
from activefly_eval_data import default_prediction_store
from activefly_eval_data import item_id


def fmt_percent(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def build_qa_outputs(data_id: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    datasets = DatasetProvider()
    predictions = default_prediction_store()
    item_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        data = datasets.load(dataset)
        if data_id:
            data = [item for item in data if str(item.get("data_id", "")) == str(data_id)]

        for item in data:
            qid = item_id(dataset, item)
            answer = str(item.get("answer", "")).upper()
            row: dict[str, Any] = {
                "dataset": dataset,
                "id": qid,
                "data_id": item.get("data_id", ""),
                "data_simorreal": item.get("data_simorreal", ""),
                "category": item.get("data_category", ""),
                "subcategory": item.get("data_subcategory", ""),
                "environment": item.get("environment", ""),
                "question_category": item.get("question_category", ""),
                "answer": answer,
            }
            for view in VIEWS:
                for model in MODELS:
                    pred = predictions.get(dataset, qid, view.name, model)
                    choice = str(pred.choice or "").upper()
                    row[f"{view.name}_{model}_choice"] = choice if choice else ""
                    row[f"{view.name}_{model}_correct"] = (
                        "" if choice not in VALID or answer not in VALID else int(choice == answer)
                    )
            item_rows.append(row)

        for view in VIEWS:
            for model in MODELS:
                total = len(data)
                answered = 0
                correct = 0
                has_source = False
                source = ""
                for item in data:
                    qid = item_id(dataset, item)
                    answer = str(item.get("answer", "")).upper()
                    pred = predictions.get(dataset, qid, view.name, model)
                    if pred.exists:
                        has_source = True
                        source = source or pred.source
                    choice = str(pred.choice or "").upper()
                    if choice in VALID:
                        answered += 1
                        if choice == answer:
                            correct += 1
                summary_rows.append({
                    "dataset": dataset,
                    "view": view.name,
                    "model": model,
                    "result_file": source if has_source else "",
                    "total": total,
                    "answered": answered if has_source else "",
                    "correct": correct if has_source else "",
                    "accuracy_percent": fmt_percent(100 * correct / answered) if answered else "",
                })

    return item_rows, summary_rows


def collect_trajectory_fluc_rows(out_dir: Path, data_id: str | None = None) -> list[dict[str, Any]]:
    path = out_dir / "trajectory_fluc.jsonl"
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = str(record.get("data_id", ""))
        if data_id and rid != str(data_id):
            continue
        for model, metrics in (record.get("metrics") or {}).items():
            if not isinstance(metrics, dict):
                continue
            rows.append({
                "source": str(path),
                "relative_path": rid,
                "data_id": rid,
                "question_id": record.get("question_id", ""),
                "model": model,
                "category": "",
                "task": "",
                "total": 1,
                "valid": 1,
                "success": metrics.get("sr", ""),
                "sr": metrics.get("sr", ""),
                "osr_valid": 1,
                "oracle_success": metrics.get("osr", ""),
                "osr": metrics.get("osr", ""),
                "ndtw_total": 1,
                "ndtw_valid": 1,
                "mean_ndtw": metrics.get("ndtw", ""),
                "ne_total": "",
                "ne_valid": "",
                "mean_ne": "",
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "eval"), help="output directory, default: ./eval")
    parser.add_argument("--data-id", default=None, help="optional trajectory data_id to collect")
    args = parser.parse_args()

    out = Path(args.out)
    item_rows, qa_summary = build_qa_outputs(args.data_id)
    if args.data_id:
        fluc_rows = collect_trajectory_fluc_rows(out, args.data_id)
    else:
        fluc_rows = collect_fluc_rows()

    item_columns = [
        "dataset", "id", "data_id", "data_simorreal", "category", "subcategory",
        "environment", "question_category", "answer",
    ]
    for view in VIEWS:
        for model in MODELS:
            item_columns.extend([f"{view.name}_{model}_choice", f"{view.name}_{model}_correct"])

    qa_summary_columns = [
        "dataset", "view", "model", "result_file", "total", "answered", "correct", "accuracy_percent",
    ]
    fluc_columns = [
        "source", "relative_path", "data_id", "question_id", "model", "category", "task", "total", "valid", "success", "sr",
        "osr_valid", "oracle_success", "osr", "ndtw_total", "ndtw_valid", "mean_ndtw",
        "ne_total", "ne_valid", "mean_ne",
    ]

    write_csv(out / "eval_items.csv", item_rows, item_columns)
    write_csv(out / "eval_qa_summary.csv", qa_summary, qa_summary_columns)
    write_csv(out / "eval_fluc_summary.csv", fluc_rows, fluc_columns)
    (out / "eval.json").write_text(
        json.dumps(
            {"items": item_rows, "qa_summary": qa_summary, "fluc_summary": fluc_rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {len(item_rows)} item rows, {len(qa_summary)} QA summary rows, {len(fluc_rows)} FLUC rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
