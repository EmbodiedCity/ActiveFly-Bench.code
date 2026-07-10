from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]

from paths import QA_DIR

RESULTS_DIR = Path(os.environ.get("ACTIVEFLY_RESULTS_DIR", str(ROOT / "eval" / "api_results")))

MODELS = ["gemini", "gpt", "qwen"]
DATASETS = ["EQA", "OBP"]
VALID = {"A", "B", "C", "D"}


@dataclass(frozen=True)
class ViewSpec:
    name: str
    suffix: str


VIEWS = [
    ViewSpec("original", ""),
    ViewSpec("openvla", "_openvla"),
    ViewSpec("pi0", "_pi0"),
]
VIEW_BY_NAME = {v.name: v for v in VIEWS}
VIEW_BY_SUFFIX = {v.suffix: v for v in VIEWS}


@dataclass
class Prediction:
    choice: str = ""
    full_response: str = ""
    source: str = ""
    exists: bool = False


def normalize_view(value: str | None) -> str:
    value = (value or "").strip().lower()
    if value in {"", "gt", "orig", "original"}:
        return "original"
    if value in {"pi0", "_pi0"}:
        return "pi0"
    if value in {"openvla", "_openvla"}:
        return "openvla"
    return value


def item_id(dataset: str, item: dict[str, Any]) -> str:
    key = "question_id" if dataset == "EQA" else "obp_id"
    return str(item.get(key, ""))


class DatasetProvider:
    def __init__(self, qa_dir: Path = QA_DIR):
        self.qa_dir = Path(qa_dir)
        self._cache: dict[str, list[dict[str, Any]]] = {}

    def load(self, dataset: str) -> list[dict[str, Any]]:
        if dataset not in self._cache:
            path = self.qa_dir / f"{dataset}.json"
            if not path.exists():
                self._cache[dataset] = []
            else:
                self._cache[dataset] = json.loads(path.read_text(encoding="utf-8"))
        return self._cache[dataset]


class PredictionStore:
    def get(self, dataset: str, qid: str, view: str, model: str) -> Prediction:
        return Prediction()


class LegacyJsonlPredictionStore(PredictionStore):
    """Adapter for API JSONL files under the configured ACTIVEFLY_RESULTS_DIR."""

    def __init__(self, results_dirs: Iterable[Path] | None = None):
        self.results_dirs = _dedupe_paths(results_dirs or default_results_dirs())
        self._cache: dict[tuple[str, str, str], tuple[dict[str, dict[str, Any]], bool, Path | None]] = {}

    def _load(self, dataset: str, model: str, suffix: str) -> tuple[dict[str, dict[str, Any]], bool, Path | None]:
        key = (dataset, model, suffix)
        if key in self._cache:
            return self._cache[key]
        records: dict[str, dict[str, Any]] = {}
        exists = False
        source: Path | None = None
        for result_dir in self.results_dirs:
            path = result_dir / f"{dataset}_{model}{suffix}.json"
            if not path.exists():
                continue
            exists = True
            if source is None:
                source = path
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = str(rec.get("question_id", ""))
                if rid and rid not in records:
                    records[rid] = rec
        self._cache[key] = (records, exists, source)
        return self._cache[key]

    def get(self, dataset: str, qid: str, view: str, model: str) -> Prediction:
        spec = VIEW_BY_NAME.get(normalize_view(view))
        if spec is None:
            return Prediction()
        records, exists, path = self._load(dataset, model, spec.suffix)
        if not exists:
            return Prediction()
        rec = records.get(str(qid), {})
        return Prediction(
            choice=str(rec.get("choice", "") or ""),
            full_response=str(rec.get("full_response", "") or ""),
            source="" if path is None else str(path),
            exists=True,
        )


class TrajectoryRecordPredictionStore(PredictionStore):
    """Adapter for future one-trajectory-at-a-time eval records.

    Supported record shapes:

    1. One prediction per line:
       {"dataset":"EQA","id":"00-0","view":"original","model":"gpt","choice":"A"}

    2. One trajectory per record:
       {
         "dataset": "EQA",
         "id": "00-0",
         "predictions": {
           "original": {"gpt": {"choice": "A", "full_response": "..."}},
           "pi0": {"qwen": "B"}
         }
       }
    """

    def __init__(self, sources: Iterable[Path]):
        self.sources = [Path(p) for p in sources]
        self._predictions: dict[tuple[str, str, str, str], Prediction] = {}
        self._loaded = False

    @classmethod
    def from_env(cls, value: str | None = None) -> "TrajectoryRecordPredictionStore | None":
        raw = value if value is not None else os.environ.get("ACTIVEFLY_TRAJECTORY_RESULTS", "")
        paths = [Path(p) for p in raw.split(os.pathsep) if p.strip()]
        return cls(paths) if paths else None

    def _iter_files(self) -> Iterable[Path]:
        for source in self.sources:
            if source.is_dir():
                yield from sorted(source.glob("*.jsonl"))
                yield from sorted(source.glob("*.json"))
            elif source.exists():
                yield source

    def _iter_records(self, path: Path) -> Iterable[dict[str, Any]]:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".jsonl":
            for line in text.splitlines():
                line = line.strip()
                if line:
                    yield json.loads(line)
            return
        payload = json.loads(text)
        if isinstance(payload, list):
            for rec in payload:
                if isinstance(rec, dict):
                    yield rec
        elif isinstance(payload, dict):
            records = payload.get("records") or payload.get("items")
            if isinstance(records, list):
                for rec in records:
                    if isinstance(rec, dict):
                        yield rec
            else:
                yield payload

    def _record_id(self, rec: dict[str, Any]) -> str:
        return str(rec.get("id") or rec.get("question_id") or rec.get("obp_id") or "")

    def _put(self, dataset: str, qid: str, view: str, model: str, value: Any, source: Path) -> None:
        if not dataset or not qid or not view or not model:
            return
        if isinstance(value, dict):
            choice = value.get("choice", "")
            full = value.get("full_response", value.get("response", ""))
        else:
            choice = value
            full = ""
        key = (dataset, qid, normalize_view(view), model)
        self._predictions[key] = Prediction(
            choice=str(choice or ""),
            full_response=str(full or ""),
            source=str(source),
            exists=True,
        )

    def _load(self) -> None:
        if self._loaded:
            return
        for path in self._iter_files():
            for rec in self._iter_records(path):
                dataset = str(rec.get("dataset", "")).upper()
                qid = self._record_id(rec)
                if "view" in rec and "model" in rec:
                    self._put(dataset, qid, str(rec.get("view", "")), str(rec.get("model", "")), rec, path)
                predictions = rec.get("predictions", {})
                if isinstance(predictions, dict):
                    for view, by_model in predictions.items():
                        if isinstance(by_model, dict):
                            for model, value in by_model.items():
                                self._put(dataset, qid, str(view), str(model), value, path)
        self._loaded = True

    def get(self, dataset: str, qid: str, view: str, model: str) -> Prediction:
        self._load()
        return self._predictions.get((dataset, str(qid), normalize_view(view), model), Prediction())


class MergedPredictionStore(PredictionStore):
    def __init__(self, stores: Iterable[PredictionStore]):
        self.stores = list(stores)

    def get(self, dataset: str, qid: str, view: str, model: str) -> Prediction:
        for store in self.stores:
            pred = store.get(dataset, qid, view, model)
            if pred.exists:
                return pred
        return Prediction()


def default_prediction_store() -> PredictionStore:
    stores: list[PredictionStore] = []
    trajectory = TrajectoryRecordPredictionStore.from_env()
    if trajectory is not None:
        stores.append(trajectory)
    stores.append(LegacyJsonlPredictionStore())
    return MergedPredictionStore(stores)


def metric_roots() -> list[Path]:
    roots = [ROOT / "eval"]
    extra = os.environ.get("ACTIVEFLY_EVAL_OUTPUT_DIRS", "")
    for item in extra.split(os.pathsep):
        if item.strip():
            roots.append(Path(item.strip()))
    return roots


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(Path(path).expanduser().resolve(strict=False))
        if key not in seen:
            seen.add(key)
            out.append(Path(path))
    return out


def default_results_dirs() -> list[Path]:
    """Results directories in priority order.

    The pipeline writes API JSONL outputs under the top-level eval/api_results
    directory by default. Legacy split-checkout results remain a read-only
    fallback for compatibility.
    """
    raw_multi = os.environ.get("ACTIVEFLY_RESULTS_DIRS", "")
    if raw_multi:
        return _dedupe_paths(Path(p) for p in raw_multi.split(os.pathsep) if p.strip())

    paths = [RESULTS_DIR]
    return _dedupe_paths(paths)


def read_text(path: Path | None) -> str:
    return "" if path is None else path.read_text(encoding="utf-8", errors="ignore")


def parse_sr(path: Path | None) -> dict[str, Any]:
    text = read_text(path)
    m = re.search(
        r"Overall Summary \(SR\).*?\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|",
        text,
        re.DOTALL,
    )
    if not m:
        return {}
    return {"total": int(m.group(1)), "valid": int(m.group(2)), "success": int(m.group(3)), "sr": m.group(4)}


def parse_osr(path: Path | None) -> dict[str, Any]:
    text = read_text(path)
    m = re.search(
        r"Overall Summary \(OSR\).*?\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|",
        text,
        re.DOTALL,
    )
    if not m:
        return {}
    return {"osr_valid": int(m.group(2)), "oracle_success": int(m.group(3)), "osr": m.group(4)}


def parse_ndtw(path: Path | None) -> dict[str, Any]:
    text = read_text(path)
    m = re.search(r"Overall Summary.*?\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|", text, re.DOTALL)
    if not m:
        return {}
    return {"ndtw_total": int(m.group(1)), "ndtw_valid": int(m.group(2)), "mean_ndtw": m.group(3)}


def parse_ne(path: Path | None) -> dict[str, Any]:
    text = read_text(path)
    m = re.search(r"Overall Summary \(NE\).*?\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|", text, re.DOTALL)
    if not m:
        return {}
    return {"ne_total": int(m.group(1)), "ne_valid": int(m.group(2)), "mean_ne": m.group(3)}


def metric_files(directory: Path) -> dict[str, Path]:
    hits: dict[str, Path] = {}
    if not directory.is_dir():
        return hits
    for path in directory.iterdir():
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.startswith("sr_"):
            hits["sr"] = path
        elif name.startswith("osr_"):
            hits["osr"] = path
        elif name.startswith("ndtw_"):
            hits["ndtw"] = path
        elif name.startswith("ne_"):
            hits["ne"] = path
    return hits


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def collect_fluc_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in metric_roots():
        if not root.is_dir():
            continue
        directories = [root, *[p for p in root.rglob("*") if p.is_dir()]]
        for directory in sorted(directories):
            files = metric_files(directory)
            if not files:
                continue
            rel = directory.relative_to(root)
            parts = [] if str(rel) == "." else list(rel.parts)
            row: dict[str, Any] = {
                "source": display_path(root),
                "relative_path": "" if not parts else "/".join(parts),
                "category": parts[0] if parts else "",
                "task": "/".join(parts[1:]) if len(parts) > 1 else "",
            }
            row.update(parse_sr(files.get("sr")))
            row.update(parse_osr(files.get("osr")))
            row.update(parse_ndtw(files.get("ndtw")))
            row.update(parse_ne(files.get("ne")))
            rows.append(row)
    return rows
