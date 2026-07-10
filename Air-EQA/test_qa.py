#!/usr/bin/env python3
"""
Run one QA sample or batch-run missing/invalid samples for a given model+dataset.

Examples:
    python3 test_qa.py --model gpt --traj pi --dataset EQA
    python3 test_qa.py --model gpt --traj pi --dataset EQA --idx 12
"""
import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

from utils import (
    SYSTEM_PROMPT,
    call_with_retry,
    extract_choice,
    image_paths,
    image_paths_openvla,
    image_paths_pi0,
    load_question,
    save_result,
)

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = Path(os.environ.get('ACTIVEFLY_RESULTS_DIR', str(SCRIPT_DIR.parent / 'eval' / 'api_results')))
VALID = {'A', 'B', 'C', 'D'}
IMAGE_PATH_FNS = {
    'gt': image_paths,
    'openvla': image_paths_openvla,
    'pi': image_paths_pi0,
}
MODEL_ENVS = {
    'gpt': ('ACTIVEFLY_GPT_MODEL', 'gpt-5.4'),
    'gemini': ('ACTIVEFLY_GEMINI_MODEL', 'gemini-2.5-pro'),
    'qwen': ('ACTIVEFLY_QWEN_MODEL', 'qwen3-vl-plus'),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=sorted(MODEL_ENVS))
    parser.add_argument('--traj', default='gt', choices=sorted(IMAGE_PATH_FNS), help='trajectory source to evaluate')
    parser.add_argument('--dataset', required=True, choices=['EQA', 'OBP'])
    parser.add_argument('--idx', type=int, default=None, help='run a single item index; omit for batch resume mode')
    parser.add_argument('--env', default=None, help='only rerun items with this environment value')
    parser.add_argument('--category', default=None, help='only rerun items with this data_category value')
    parser.add_argument('--rewrite', action='store_true', help='rerun even if already has valid answer; latest result wins')
    return parser.parse_args()


def result_tag(model: str, traj: str) -> str:
    return model if traj == 'gt' else f'{model}_{traj}'


def build_openai_content(paths: list[Path], question: str) -> list[dict]:
    return [
        {
            'type': 'image_url',
            'image_url': {'url': f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"},
        }
        for path in paths
    ] + [{'type': 'text', 'text': question}]


def run_gpt(paths: list[Path], question: str) -> tuple[str, str]:
    from openai import OpenAI

    api_key = os.environ.get('OPENAI_API_KEY') or os.environ.get('ACTIVEFLY_OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('Set OPENAI_API_KEY or ACTIVEFLY_OPENAI_API_KEY before running GPT tests.')
    base_url = os.environ.get('OPENAI_BASE_URL') or os.environ.get('ACTIVEFLY_OPENAI_BASE_URL')
    client_kwargs = {'api_key': api_key}
    if base_url:
        client_kwargs['base_url'] = base_url
    client = OpenAI(**client_kwargs)
    model = os.environ.get(*MODEL_ENVS['gpt'])
    response, err = call_with_retry(lambda: client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': build_openai_content(paths, question)},
        ],
    ))
    if err:
        return f'Error: {err}', str(err)
    text = response.choices[0].message.content
    return extract_choice(text), text


def run_gemini(paths: list[Path], question: str) -> tuple[str, str]:
    from google import genai
    from google.genai import types

    api_key = (
        os.environ.get('GEMINI_API_KEY')
        or os.environ.get('GOOGLE_API_KEY')
        or os.environ.get('ACTIVEFLY_GEMINI_API_KEY')
    )
    if not api_key:
        raise RuntimeError('Set GEMINI_API_KEY, GOOGLE_API_KEY, or ACTIVEFLY_GEMINI_API_KEY before running Gemini tests.')
    base_url = os.environ.get('GOOGLE_GEMINI_BASE_URL') or os.environ.get('ACTIVEFLY_GEMINI_BASE_URL')
    client_kwargs = {'api_key': api_key}
    if base_url:
        client_kwargs['http_options'] = types.HttpOptions(base_url=base_url)
    client = genai.Client(**client_kwargs)
    model = os.environ.get(*MODEL_ENVS['gemini'])
    contents = [types.Part.from_bytes(data=path.read_bytes(), mime_type='image/png') for path in paths] + [question]
    response, err = call_with_retry(lambda: client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    ))
    if err:
        return f'Error: {err}', str(err)
    return extract_choice(response.text), response.text


def run_qwen(paths: list[Path], question: str) -> tuple[str, str]:
    import dashscope
    from dashscope import MultiModalConversation

    api_key = (
        os.environ.get('DASHSCOPE_API_KEY')
        or os.environ.get('QWEN_API_KEY')
        or os.environ.get('ACTIVEFLY_QWEN_API_KEY')
    )
    if not api_key:
        raise RuntimeError('Set DASHSCOPE_API_KEY, QWEN_API_KEY, or ACTIVEFLY_QWEN_API_KEY before running Qwen tests.')
    dashscope.api_key = api_key
    base_url = (
        os.environ.get('DASHSCOPE_BASE_URL')
        or os.environ.get('QWEN_BASE_URL')
        or os.environ.get('ACTIVEFLY_QWEN_BASE_URL')
    )
    if base_url:
        for attr in ('base_http_api_url', 'base_api_url'):
            if hasattr(dashscope, attr):
                setattr(dashscope, attr, base_url)
    model = os.environ.get(*MODEL_ENVS['qwen'])
    user_content = [{'image': path.resolve().as_uri()} for path in paths] + [{'text': question}]
    response, err = call_with_retry(lambda: MultiModalConversation.call(
        model=model,
        messages=[
            {'role': 'system', 'content': [{'text': SYSTEM_PROMPT}]},
            {'role': 'user', 'content': user_content},
        ],
    ))
    if err:
        return f'Error: {err}', str(err)
    if response.status_code != 200:
        msg = f'Error {response.code}: {response.message}'
        return msg, msg
    text = response.output.choices[0].message.content[0]['text']
    return extract_choice(text), text


RUNNERS = {
    'gpt': run_gpt,
    'gemini': run_gemini,
    'qwen': run_qwen,
}


def load_results(path: Path, latest_wins: bool = False) -> dict:
    results = {}
    if not path.exists():
        return results
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            qid = rec.get('question_id')
            if qid is None:
                continue
            if latest_wins:
                results[qid] = rec
            else:
                prev = results.get(qid)
                if prev is None:
                    results[qid] = rec
                elif rec.get('choice') in VALID and prev.get('choice') not in VALID:
                    results[qid] = rec
        except Exception:
            pass
    return results


def rewrite_ordered(path: Path, data: list, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        for item in data:
            qid = item.get('question_id') or item.get('obp_id')
            if qid in results:
                handle.write(json.dumps(results[qid], ensure_ascii=False) + '\n')


def run_single(model: str, traj: str, dataset: str, idx: int) -> None:
    item = load_question(dataset, idx)
    paths = IMAGE_PATH_FNS[traj](item, dataset)
    choice, full_response = RUNNERS[model](paths, item['question'])
    print(save_result(dataset, result_tag(model, traj), item, choice, full_response))


def run_batch(args: argparse.Namespace) -> None:
    tag = result_tag(args.model, args.traj)
    data = json.loads((SCRIPT_DIR / f'{args.dataset}.json').read_text(encoding='utf-8'))
    result_path = RESULTS_DIR / f'{args.dataset}_{tag}.json'

    existing = load_results(result_path)
    needed = []
    for idx, item in enumerate(data):
        if args.env and item.get('environment') != args.env:
            continue
        if args.category and item.get('data_category') != args.category:
            continue
        qid = item.get('question_id') or item.get('obp_id')
        if args.rewrite or qid not in existing or existing[qid].get('choice') not in VALID:
            needed.append(idx)

    total = len(data)
    print(f'=== {tag} | {args.dataset}: {len(needed)} to run / {total} total ===')

    for pos, idx in enumerate(needed, start=1):
        print(f'[{args.dataset} {pos}/{len(needed)} (idx={idx})] ', end='', flush=True)
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / 'test_qa.py'),
                '--model',
                args.model,
                '--traj',
                args.traj,
                '--dataset',
                args.dataset,
                '--idx',
                str(idx),
            ],
            check=False,
        )

    updated = load_results(result_path, latest_wins=args.rewrite)
    rewrite_ordered(result_path, data, updated)

    valid_count = sum(1 for record in updated.values() if record.get('choice') in VALID)
    print(f'\n=== {tag} | {args.dataset} done: {valid_count}/{total} valid, file rewritten in order ===')


def main() -> None:
    args = parse_args()
    if args.idx is not None:
        run_single(args.model, args.traj, args.dataset, args.idx)
    else:
        run_batch(args)


if __name__ == '__main__':
    main()
