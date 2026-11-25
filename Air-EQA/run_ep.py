"""Run OBP, policy rollout and post-rollout EQA for one simulated scene."""
import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from test_qa import RUNNERS
from utils import QA_DIR, _subsample, image_paths


ROOT = Path(__file__).resolve().parents[1]
CLIENT = {
    "openvla": ROOT / "FLUC-OpenVLA/eval/run_act_client.py",
    "pi": ROOT / "FLUC-OpenPi/eval/run_rollout.py",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def planning_input(plan, question):
    # Some released EQA rows start their options directly with "A.".
    stem = re.split(r"\bChoices\s*:|(?<!\S)A[.)]\s+", question,
                    maxsplit=1, flags=re.IGNORECASE)[0].strip()
    header, _, choices = plan["question"].partition("Choices:")
    source = re.fullmatch(r'What action is needed to answer the question:\s*"(.*)"',
                          header.strip(), re.DOTALL)
    if not stem or source is None:
        raise ValueError("Missing EQA stem or unrecognized OBP question header")
    parts = re.split(r"(?<!\S)([A-D])[.)]\s+", choices)
    labels, texts = parts[1::2], [text.strip() for text in parts[2::2]]
    if labels != list("ABCD") or not all(texts):
        raise ValueError("Expected one nonempty instruction for each of A, B, C, D")
    instructions = dict(zip(labels, texts))
    reference = plan["answer"].strip().upper()
    if stem != source[1].strip() or texts.count(instructions[reference]) != 1:
        reference = None
    prompt = f'What action is needed to answer the question: "{stem}"\n\nChoices:{choices}'
    return prompt, instructions, reference


def oracle_success(log, trajectory):
    # GT deltas use z-up and [yaw, pitch]. Client logs use local z-down
    # positions and [pitch, yaw, roll]. See Appendix A.5.1, Eq. (7).
    deltas = log["relative_action"]
    if not deltas or any(len(row) != 5 for row in deltas):
        raise ValueError("Missing or invalid reference trajectory.")
    x, y, z, yaw, pitch = map(sum, zip(*deltas))
    for step in trajectory:
        xyz, rot = step["state"]
        angle = abs(math.remainder(rot[0] - pitch, 360)) + abs(math.remainder(rot[1] - yaw, 360))
        if math.dist(xyz, (x, y, -z)) < 3 and angle < 10:
            return True
    return False


def run_episode(args, plan, q, folder):
    data_id = plan["data_id"]
    folder.mkdir(parents=True)
    result = dict(data_id=data_id, question_id=q["question_id"], environment=args.scene,
                  model=args.model, policy=args.policy, obp_id=plan["obp_id"])
    try:
        prompt, instructions, reference = planning_input(plan, q["question"])
    except ValueError as exc:
        result.update(status="invalid_input", reason=str(exc), ep_success=None)
        write_json(folder / "result.json", result)
        return result
    initial = image_paths(plan, "OBP")
    if not initial:
        raise FileNotFoundError(f"Missing initial frame for {data_id}")
    ask = RUNNERS[args.model]
    prediction, _ = ask(initial, prompt)
    if prediction not in instructions:
        raise RuntimeError(f"Invalid OBP response for {q['question_id']}: {prediction!r}")
    instruction = instructions[prediction]
    log = read_json(initial[0].parent / "log.json")
    log.update(instruction=instruction, instruction_unified=instruction)
    inputs = folder / "input" / data_id
    inputs.mkdir(parents=True)
    write_json(inputs / "log.json", log)

    # Limit discovery to this log. Never reuse another planner's rollout.
    output = folder / "rollout"
    subprocess.run([sys.executable, str(CLIENT[args.policy]), *args.client_args, "--require_complete",
                    "-r", str(inputs), "-o", str(output), "--server_ip", args.server_ip,
                    "--server_port", str(args.server_port)], check=True, cwd=ROOT)
    episode = output / data_id
    trajectory = read_json(episode / f"{data_id}.json")
    frames = sorted((episode / "frames").glob("*.png"), key=lambda p: int(p.stem))
    if not trajectory or not frames:
        raise RuntimeError(f"No usable rollout for {data_id}")
    sampled = _subsample(frames, 16)
    oracle = oracle_success(log, trajectory)
    answer, _ = ask(sampled, q["question"])
    if answer not in {"A", "B", "C", "D"}:
        raise RuntimeError(f"Invalid EQA response for {q['question_id']}: {answer!r}")
    correct = answer == q["answer"].strip().upper()
    obp_ok = prediction == reference if reference is not None else None
    result.update(status="completed", obp_prompt=prompt, obp_choice=prediction,
                  obp_reference=reference, selected_instruction=instruction,
                  obp_correct=obp_ok, fluc_oracle_success=oracle,
                  eqa_frames=[p.name for p in sampled], eqa_choice=answer,
                  eqa_reference=q["answer"], eqa_correct=correct,
                  ep_success=int(obp_ok and oracle and correct) if reference is not None else None)
    write_json(folder / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=RUNNERS, required=True)
    parser.add_argument("--policy", choices=CLIENT, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--server-ip", required=True)
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-id")
    parser.add_argument("--eqa-idx", type=int)
    parser.add_argument("client_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.client_args[:1] == ["--"]:
        args.client_args = args.client_args[1:]
    if any(arg.split("=")[0] in {"-r", "--log_root", "-o", "--images_dir"} for arg in args.client_args):
        parser.error("EP owns the rollout input and output paths.")
    questions = read_json(QA_DIR / "EQA.json")
    if args.eqa_idx is not None:
        if not 0 <= args.eqa_idx < len(questions):
            parser.error("EQA index is outside the dataset.")
        questions = [questions[args.eqa_idx]]
    questions = [q for q in questions if q["environment"] == args.scene
                 and q["data_simorreal"] == "sim"
                 and (args.data_id is None or q["data_id"] == args.data_id)]
    if not questions:
        parser.error("No simulated EQA samples match this selection.")
    plans = defaultdict(list)
    for plan in read_json(QA_DIR / "OBP.json"):
        if plan["environment"] == args.scene and plan["data_simorreal"] == "sim":
            plans[plan["data_id"]].append(plan)
    args.output_root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(dir=args.output_root, prefix=f"{args.scene}-{args.model}-{args.policy}-")).resolve()
    print(f"EP results: {run}", flush=True)
    scores = []
    for q in questions:
        data_id = q["data_id"]
        if len(plans[data_id]) != 1:
            raise ValueError(f"Expected one trajectory-linked OBP record for {data_id}")
        scores.append(run_episode(args, plans[data_id][0], q, run / data_id / q["question_id"]))
    scored = [row["ep_success"] for row in scores if row["ep_success"] is not None]
    successes = sum(scored)
    completed = sum(row["status"] == "completed" for row in scores)
    write_json(run / "summary.json", dict(successes=successes, samples=len(scores),
                                          completed=completed, invalid_inputs=len(scores) - completed,
                                          scored=len(scored), unlabelled=completed - len(scored),
                                          ep_success_rate=successes / len(scored) if scored else None))
    print(f"EP successes: {successes} of {len(scored)} scorable questions ({len(scores)} selected)")


if __name__ == "__main__":
    main()
