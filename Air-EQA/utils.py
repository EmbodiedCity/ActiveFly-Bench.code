import argparse
import base64
import json
import os
import random
import re
import sys
import threading
import time
from pathlib import Path

from PIL import Image

ACTIVEFLY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ACTIVEFLY_ROOT / "utils"))
from paths import FLUC_SIM_DIR, OPENVLA_DIR, PI0_DIR, QA_DIR, FLUC_REAL_DIR, ROLLOUT_DIR

_REAL_MAP: dict | None = None


def _real_data_map() -> dict:
    global _REAL_MAP
    if _REAL_MAP is None:
        _REAL_MAP = {path.parent.name: path.parent for path in FLUC_REAL_DIR.rglob('000000.png')}
    return _REAL_MAP


SYSTEM_PROMPT = """You are an expert visual QA assistant. Look closely at the provided images and answer the question.
Your task is to determine the correct choice based on the given images.
Provide your final choice among A, B, C, D.

First, think step-by-step and provide your reasoning (Chain of Thought).
Then, you MUST output your final choice wrapped between <answer>[Your Choice]</answer> tags. Do not include any other punctuation, explanations, or extra text inside the tags.
The final choice MUST be exactly one of: \"A\", \"B\", \"C\", \"D\".

Format:
Reasoning: [your step-by-step analysis here]
<answer>A</answer>"""


def extract_choice(text: str) -> str:
    if not text or text.startswith('Error:') or text in ('API_NO_RESPONSE', 'Error: NOT_CONFIGURED'):
        return text
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.IGNORECASE | re.DOTALL)
    answer = match.group(1).strip().upper() if match else text.strip().upper()
    for letter in ('A', 'B', 'C', 'D'):
        if answer in (letter, f'{letter}.', f'{letter})'):
            return letter
    return answer


IO_LOCK = threading.Lock()
_IMAGE_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


class Logger:
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def get_cached_images(image_path: str | Path):
    key = str(image_path)
    with _CACHE_LOCK:
        if key in _IMAGE_CACHE:
            return _IMAGE_CACHE[key]
    img = Image.open(image_path)
    img.load()
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    with _CACHE_LOCK:
        _IMAGE_CACHE[key] = (img, b64)
    return img, b64


def call_with_retry(fn, *, max_retries: int = 3, delay: float = 5.0):
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn(), None
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(delay)
    return None, last_exc


def is_answer_valid(ai_answer: str) -> bool:
    return bool(ai_answer) and not ai_answer.startswith('Error:') and ai_answer != 'API_NO_RESPONSE'


def is_answer_correct(ai_answer: str, gt_answer: str) -> bool:
    return is_answer_valid(ai_answer) and ai_answer.upper() == gt_answer.strip().upper()


def all_answered_correctly(model_answers: dict[str, str], gt_answer: str) -> bool:
    return bool(model_answers) and all(is_answer_correct(ans, gt_answer) for ans in model_answers.values())


def write_log(handle, q_id, question: str, ai_answer: str, gt_answer: str):
    handle.write(f'Question ID: {q_id}\n')
    handle.write(f'Question: {question}\n')
    handle.write(f'AI Answer: {ai_answer}\n')
    handle.write(f'Ground Truth: {gt_answer}\n')
    handle.write('-' * 40 + '\n')
    handle.flush()


def build_data_id_map(dataset_root: str | Path) -> dict[str, str]:
    print('正在扫描数据集目录，建立数据映射...')
    data_map: dict[str, str] = {}
    for root, _, files in os.walk(dataset_root):
        if '0.png' not in files:
            continue
        data_map[os.path.basename(root)] = root
        log_path = os.path.join(root, 'log.json')
        if os.path.exists(log_path):
            try:
                log = json.loads(Path(log_path).read_text(encoding='utf-8'))
                if 'id' in log:
                    data_map[log['id']] = root
            except Exception as exc:
                print(f'读取 {log_path} 时出错: {exc}')
    print(f'扫描完成，共找到 {len(data_map)} 个唯一的数据项映射。\n')
    return data_map


def load_question(dataset='EQA', idx=None):
    path = Path(__file__).parent / f'{dataset}.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    return data[idx if idx is not None else random.randrange(len(data))]


def _subsample(paths: list, n: int = 16) -> list:
    if len(paths) <= n:
        return paths
    step = (len(paths) - 1) / (n - 1)
    return [paths[round(i * step)] for i in range(n)]


def image_paths(item, dataset='EQA'):
    cat, env, data_id = item['data_category'], item['environment'], item['data_id']
    if item.get('data_simorreal') == 'real':
        folder = _real_data_map().get(data_id)
        if folder is None:
            raise FileNotFoundError(f'real frames not found for data_id={data_id}')
    elif cat == 'action':
        sub = item['data_subcategory']
        base = FLUC_SIM_DIR / 'action' / 'action_raw' / 'action_all_mix_test' / env
        candidates = [base / sub / data_id, base / sub.replace(' ', '_') / data_id]
        folder = next((path for path in candidates if path.is_dir()), None)
        if folder is None:
            raise FileNotFoundError(f'action frames not found for data_id={data_id}, sub={sub}')
    else:
        folder = FLUC_SIM_DIR / cat / 'test' / env / data_id
    paths = _subsample(sorted(folder.glob('*.png'), key=lambda path: int(path.stem)))
    return paths[:1] if dataset == 'OBP' else paths


def _rollout_frame_dir(root: Path, item: dict, label: str) -> Path:
    data_id = item['data_id']

    # Unified rollout layout for both OpenVLA and OpenPi:
    #   <rollout_root>/<data_id>/frames/*.png
    folder = root / data_id / 'frames'
    if folder.is_dir():
        return folder

    # Backward-compatible fallback for legacy OpenVLA outputs.
    if label == 'openvla':
        cat, env = item['data_category'], item['environment']
        if cat == 'action':
            action = item['data_subcategory']
            candidates = [
                root / cat / 'cotraining_tpz' / env / action / data_id / 'frames',
                root / cat / 'cotraining_tpz' / env / action.replace(' ', '_') / data_id / 'frames',
            ]
        else:
            candidates = [root / cat / 'cotraining_tpz' / env / data_id / 'frames']
        legacy = next((path for path in candidates if path.is_dir()), None)
        if legacy is not None:
            return legacy

    raise FileNotFoundError(f'{label} frames not found for data_id={data_id}')


def image_paths_pi0(item, dataset='EQA'):
    folder = _rollout_frame_dir(PI0_DIR, item, 'pi0')
    paths = _subsample(sorted(folder.glob('*.png'), key=lambda path: int(path.stem)))
    return paths[:1] if dataset == 'OBP' else paths


def image_paths_openvla(item, dataset='EQA'):
    folder = _rollout_frame_dir(OPENVLA_DIR, item, 'openvla')
    paths = _subsample(sorted(folder.glob('*.png'), key=lambda path: int(path.stem)))
    return paths[:1] if dataset == 'OBP' else paths


def parse_args(include_model: bool = False):
    parser = argparse.ArgumentParser()
    if include_model:
        parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', choices=['EQA', 'OBP'], default='EQA')
    parser.add_argument('--idx', type=int, default=0)
    return parser.parse_args()


def save_result(dataset: str, model: str, item: dict, choice: str, full_response: str) -> str:
    record = {
        'question_id': item.get('question_id') or item.get('obp_id'),
        'choice': choice,
        'full_response': full_response,
    }
    results_dir = Path(os.environ.get('ACTIVEFLY_RESULTS_DIR', str(ACTIVEFLY_ROOT / 'eval' / 'api_results')))
    out_path = results_dir / f'{dataset}_{model}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return json.dumps(record, ensure_ascii=False)
