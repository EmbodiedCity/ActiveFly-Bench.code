#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
from paths import QA_DIR


def main():
    parser = argparse.ArgumentParser(description="List ActiveFly dataset indices for one trajectory data_id.")
    parser.add_argument("--dataset", choices=["EQA", "OBP"], required=True)
    parser.add_argument("--data-id", required=True)
    args = parser.parse_args()

    data = json.loads((QA_DIR / f"{args.dataset}.json").read_text(encoding="utf-8"))
    id_key = "question_id" if args.dataset == "EQA" else "obp_id"
    for idx, item in enumerate(data):
        if str(item.get("data_id", "")) == str(args.data_id):
            print(f"{idx}\t{item.get(id_key, '')}")


if __name__ == "__main__":
    main()
