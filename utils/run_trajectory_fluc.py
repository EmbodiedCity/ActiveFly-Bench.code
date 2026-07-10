#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
from paths import QA_DIR


def main():
    parser = argparse.ArgumentParser(description="Compute FLUC metrics for one ActiveFly trajectory.")
    parser.add_argument("--data-id", required=True)
    parser.add_argument("--out", default=str(ROOT / "eval" / "trajectory_fluc.jsonl"))
    args = parser.parse_args()

    eqa = json.loads((QA_DIR / "EQA.json").read_text(encoding="utf-8"))
    match = next((item for item in eqa if str(item.get("data_id", "")) == str(args.data_id)), None)
    if match is None:
        raise SystemExit(f"No EQA trajectory found for data_id={args.data_id}")

    sys.path.insert(0, str(QA_DIR))
    import metric_qa

    metrics = {
        "openvla": metric_qa.fluc_metrics(match, "_openvla"),
        "pi0": metric_qa.fluc_metrics(match, "_pi0"),
    }
    record = {
        "dataset": "FLUC",
        "data_id": args.data_id,
        "question_id": match.get("question_id"),
        "metrics": metrics,
        "created_at": time.time(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
