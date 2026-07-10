"""Real-time OpenVLA action server for ActiveFly 5-DoF pose control."""

import argparse
import base64
import logging
from io import BytesIO
from pathlib import Path
import os
import sys
import time

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

sys.path.append(str(Path(__file__).resolve().parents[1]))

log = logging.getLogger(__name__)


class OpenVLAActionAgent:
    def __init__(self, cfg: argparse.Namespace):
        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg.gpu_id}")
        self.model_path = cfg.model_path
        self.unnorm_key = cfg.unnorm_key
        self.do_sample = cfg.do_sample
        self.port = cfg.http_port
        self.host = cfg.host

        log.info("Loading model from %s", self.model_path)
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self) -> None:
        @self.app.route("/predict", methods=["POST"])
        def predict():
            try:
                data = request.json
                img_bytes = base64.b64decode(data["image"])
                image = Image.open(BytesIO(img_bytes))
                instruction = data["instr"]
                proprio = np.array(data["proprio"], dtype=np.float32)
                proprio_str = ",".join(str(round(x, 1)) for x in proprio)
                prompt = f"In: Current State: {proprio_str}, What action should the uav take to {instruction}?\nOut:"
                log.info("Prompt: %s", prompt)

                start_time = time.time()
                with torch.inference_mode():
                    pred_action = self.act(image, prompt)
                log.info("Inference time: %.3f seconds", time.time() - start_time)

                pred_action = pred_action[None, :]
                pred_action_ori = pred_action.copy()

                current_yaw = np.deg2rad(proprio[-2])
                current_pit = np.deg2rad(proprio[-1])
                current_pos = proprio[0:3]
                cos_yaw = np.cos(current_yaw)
                sin_yaw = np.sin(current_yaw)
                rotation = np.array(
                    [
                        [cos_yaw, -sin_yaw, 0],
                        [sin_yaw, cos_yaw, 0],
                        [0, 0, 1],
                    ]
                )

                pred_action[0, 0:3] = rotation @ pred_action[0, 0:3]
                pred_action[0, 0:3] = current_pos + pred_action[0, 0:3]
                pred_action[0, -2] = pred_action[0, -2] + current_yaw
                pred_action[0, -1] = pred_action[0, -1] + current_pit

                return jsonify(
                    {
                        "status": "success",
                        "action": pred_action.tolist(),
                        "action_ori": pred_action_ori.tolist(),
                        "message": "Action generated successfully",
                    }
                )
            except Exception as exc:
                import traceback

                traceback.print_exc()
                return jsonify({"status": "error", "message": str(exc) + traceback.format_exc()}), 500

        @self.app.route("/reset", methods=["POST"])
        def reset():
            return jsonify({"status": "success", "message": "No-op reset completed"})

    def run(self) -> None:
        log.info("Starting HTTP server on %s:%s", self.host, self.port)
        self.app.run(host=self.host, port=self.port)

    def act(self, image: Image.Image, prompt: str):
        inputs = self.processor(prompt, image)
        inputs = inputs.to(self.device, dtype=torch.bfloat16)
        return self.model.predict_action(**inputs, unnorm_key=self.unnorm_key, do_sample=self.do_sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve an ActiveFly OpenVLA checkpoint over HTTP.")
    parser.add_argument("--model-path", required=True, help="Path to the merged OpenVLA checkpoint directory.")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--http-port", type=int, default=5007, help="HTTP port for the /predict server.")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host.")
    parser.add_argument("--unnorm-key", default="activefly", help="Normalization stats key stored in the checkpoint.")
    parser.add_argument("--do-sample", action="store_true", help="Enable stochastic action sampling.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    agent = OpenVLAActionAgent(args)
    agent.run()


if __name__ == "__main__":
    main()
