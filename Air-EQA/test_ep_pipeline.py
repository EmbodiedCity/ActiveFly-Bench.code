import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import run_ep


class PredictedPlanTest(unittest.TestCase):
    def test_wrong_plan_is_executed_and_its_frames_still_reach_eqa(self):
        for policy in run_ep.CLIENT:
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                gt = root / "gt"
                gt.mkdir()
                initial = gt / "0.png"
                initial.touch()
                source = {"instruction": "reference", "relative_action": [[100, 0, 0, 0, 0]]}
                run_ep.write_json(gt / "log.json", source)
                plan = dict(data_id="example", obp_id="plan", answer="A",
                            environment="citypark", data_simorreal="sim",
                            question='What action is needed to answer the question: "Color?"\n\nChoices:\nA. reference\nB. fly left\nC. look up\nD. descend')
                questions = [
                    dict(question_id="color", question="Color? Choices: A. red B. blue", answer="B"),
                    dict(question_id="size", question="Size? A. big B. small", answer="A"),
                ]
                for q in questions:
                    q.update(data_id="example", data_simorreal="sim", environment="citypark")
                run_ep.write_json(root / "EQA.json", questions)
                run_ep.write_json(root / "OBP.json", [plan])
                output = root / "runs"
                argv = ["run_ep.py", "--model", "qwen", "--policy", policy, "--scene", "citypark",
                        "--server-ip", "localhost", "--server-port", "5007", "--output-root", str(output)]
                commands = []

                def rollout(cmd, **kwargs):
                    self.assertEqual(Path(cmd[1]), run_ep.CLIENT[policy])
                    log = run_ep.read_json(Path(cmd[cmd.index("-r") + 1]) / "log.json")
                    commands.append(log["instruction"])
                    self.assertEqual(log["instruction_unified"], log["instruction"])
                    out = Path(cmd[cmd.index("-o") + 1]) / "example"
                    frames = out / "frames"
                    frames.mkdir(parents=True)
                    (frames / "000001.png").touch()
                    run_ep.write_json(out / "example.json", [{"state": [[0, 0, 0], [0, 0, 0]]}])

                ask = Mock(side_effect=[("B", ""), ("B", ""), ("C", ""), ("A", "")])
                with patch.object(run_ep, "image_paths", return_value=[initial]), \
                     patch.dict(run_ep.RUNNERS, qwen=ask), patch.object(run_ep, "QA_DIR", root), \
                     patch.object(sys, "argv", argv), patch.object(run_ep.subprocess, "run", side_effect=rollout):
                    run_ep.main()
                run, = output.iterdir()
                self.assertEqual(commands, ["fly left", "look up"])
                for i, q in enumerate(questions):
                    folder = run / "example" / q["question_id"]
                    expected_prompt = plan["question"].replace("Color?", ["Color?", "Size?"][i])
                    image = folder / "rollout/example/frames/000001.png"
                    self.assertEqual(ask.call_args_list[2*i].args, ([initial], expected_prompt))
                    self.assertEqual(ask.call_args_list[2*i+1].args, ([image], q["question"]))
                    result = run_ep.read_json(folder / "result.json")
                    self.assertIs(result["obp_correct"], [False, None][i])
                    self.assertEqual(result["ep_success"], [0, None][i])
                    self.assertTrue(result["eqa_correct"])
                summary = run_ep.read_json(run / "summary.json")
                self.assertEqual((summary["samples"], summary["completed"], summary["scored"]), (2, 2, 1))
                self.assertEqual(run_ep.read_json(gt / "log.json"), source)

    def test_duplicate_label_is_invalid(self):
        plan = {"question": 'What action is needed to answer the question: "Color?"\nChoices: '
                            'A. left B. right C. up D. down D. forward', "answer": "D"}
        with self.assertRaisesRegex(ValueError, "one nonempty instruction"):
            run_ep.planning_input(plan, "Color? Choices: A. red B. blue")


if __name__ == "__main__":
    unittest.main()
