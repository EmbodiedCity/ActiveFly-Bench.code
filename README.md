# ActiveFly-Bench: Aligning Embodied Question Answering with Vision-Language-Action for Aerial Embodied Perception

## TODO list
- [x] Dataset Released
  - [x] Simulated Datasets
  - [x] Real-world Datasets
- [x] Pretrained Weights Released
- [x] Simulation Deployment Released
  - [x] OpenVLA-based Agent
  - [x] OpenPi-based Agent
- [x] Real-world Deployment Released
  - [x] OpenVLA-based Agent
  - [x] OpenPi-based Agent

## 1. Environment Setup
### OpenVLA-based Agent

Use this environment for OpenVLA training, rollout, and FLUC evaluation.

```bash
conda create -n activefly-openvla python=3.11 -y
conda activate activefly-openvla

pip install -r requirements-openvla.txt
```

Install `flash-attn` after the main requirements are installed.

```bash
pip install packaging ninja
pip install flash-attn==2.5.5 --no-build-isolation
```

`flash-attn` can also be installed from a [prebuilt wheel](https://github.com/Dao-AILab/flash-attention/releases?page=5).

### OpenPi-based Agent

Use this environment for OpenPi training, rollout, and FLUC evaluation.

```bash
conda create -n activefly-pi python=3.11 -y
conda activate activefly-pi

pip install -r requirements-pi.txt
```

### Dataset

Download the packaged ActiveFly train/test archives from [Hugging Face Datasets](https://huggingface.co/datasets/EmbodiedCity/ActiveFly-Bench/tree/main/datasets).

### Simulated Environments

Download the simulation assets from [Hugging Face](https://huggingface.co/datasets/EmbodiedCity/ActiveFly-Bench/tree/main/Sim_Envs) or [EmbodiedCity](https://github.com/tsinghua-fib-lab/EmbodiedCity).

Then see [Configuration](#2-configuration) to set API keys and external dataset paths.

## 2. Configuration

Configure environment variables before running evaluation, the pipeline, or the [Web UI](#5-web-ui). Use `export` in your shell, or place variables in a local `.env` file (loaded by `run_pipeline.sh` and `run_eval.sh`).


### Dataset Paths Config (set on your machine)

GT and rollout frames are **not** in the repo. Point these to where you downloaded ActiveFly on your machine:

| Variable | Purpose |
|----------|---------|
| `ACTIVEFLY_QA_DIR` | Air-EQA / OBP metadata directory (default: `./Air-EQA`) |
| `ACTIVEFLY_FLUC_SIM_DIR` | GT trajectory frames (sim) |
| `ACTIVEFLY_FLUC_REAL_DIR` | GT trajectory frames (real-world) |
| `ACTIVEFLY_ROLLOUT_DIR` | Rollout frame root (`pi/`, `openvla/` subdirs) |
| `ACTIVEFLY_PI0_DIR` | Optional Pi0 rollout output override (default: `$ACTIVEFLY_ROLLOUT_DIR/pi`) |
| `ACTIVEFLY_OPENVLA_DIR` | Optional OpenVLA rollout output override (default: `$ACTIVEFLY_ROLLOUT_DIR/openvla`) |

Example:

```bash
export ACTIVEFLY_FLUC_SIM_DIR=<path_to_ActiveFly_on_your_machine>
export ACTIVEFLY_FLUC_REAL_DIR=<path_to_ActiveFly_on_your_machine>/realworld_indoor
export ACTIVEFLY_PI0_DIR=<pi0_rollout_or_checkpoint_dir>
export ACTIVEFLY_OPENVLA_DIR=<openvla_rollout_dir>
export ACTIVEFLY_ROLLOUT_DIR=<rollout_root>   # optional; defaults to pi/ and openvla/ under it
```

Create a local `.env` with the variables above (and API keys), then run `run_pipeline.sh` / `run_eval.sh` (they load `.env` automatically). `pipeline_web.py` also loads `.env` on start.

### Output Directory (Defaults)

These paths are default output directory in this repo:

| Path | Purpose |
|------|---------|
| `eval/api_results/` | VLM answer JSONL files |
| `eval/` | Collected summary tables (`eval_items.csv`, `eval.json`, …) |
| `eval/pipeline_web/` | Static debug pages built by `run_eval.sh` |


### API Keys

`Air-EQA/test_qa.py` reads API credentials from environment variables:

- OpenAI: `OPENAI_API_KEY` or `ACTIVEFLY_OPENAI_API_KEY`
- Gemini: `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `ACTIVEFLY_GEMINI_API_KEY`
- Qwen: `DASHSCOPE_API_KEY`, `QWEN_API_KEY`, or `ACTIVEFLY_QWEN_API_KEY`

```bash
export DASHSCOPE_API_KEY=<your_qwen_api_key>
```

## 3. ActiveFly Workflow

### 3.1 Air-EQA and OBP Evaluation

Run the following command to evaluate a VLM to answer EQA or OBP from GT or rollout frames.

```bash
python Air-EQA/test_qa.py --model <gpt|gemini|qwen> --traj <gt|openvla|pi> --dataset <EQA|OBP>
```

Notes:

- `--traj gt` reads the ground-truth trajectory frames.
- `--traj openvla` reads rollout frames under `openvla/` in your configured rollout directory.
- `--traj pi` reads rollout frames under `pi/` in your configured rollout directory.
- VLM answers are written to `eval/api_results/` by default. See [Configuration](#2-configuration) for external dataset paths.

Example:

```bash
python Air-EQA/test_qa.py --model qwen --traj openvla --dataset EQA
python Air-EQA/test_qa.py --model qwen --traj pi --dataset OBP
```

### 3.2 FLUC Evaluation

Download [OpenVLA](https://huggingface.co/datasets/EmbodiedCity/ActiveFly-Bench/tree/main/weights/openvla) or [OpenPi](https://huggingface.co/datasets/EmbodiedCity/ActiveFly-Bench/tree/main/weights/openpi) weights.  Run the server and client in separate terminals.

#### OpenVLA

Server:

Start the OpenVLA policy server before launching the FLUC rollout client.

```bash
conda activate activefly-openvla
cd FLUC-OpenVLA/run

python vla-scripts/run_act_server.py \
  --model-path <model_checkpoint> \
  --http-port 5007 \
  --gpu-id 0 \
  --unnorm-key sim
```

Client:

Run the rollout client after the OpenVLA server is already serving requests.

```bash
conda activate activefly-openvla
cd FLUC-OpenVLA/eval

python run_act_client.py \
  -r <data_root>/ActiveFly_Test/<split> \
  -o ../outputs/openvla_activefly_eval \
  --server_port 5007 \
  --instruction_type instruction
```

Metrics:

Run the shared metric script after the OpenVLA rollout.

```bash
python ../../scripts/metric.py \
  <data_root>/ActiveFly_Test/<split> \
  ../outputs/openvla_activefly_eval \
  --metric all \
  --output-dir ./eval_outputs
```

#### OpenPi

Server:

Start the OpenPi policy server before launching the FLUC rollout client.

```bash
conda activate activefly-pi
cd FLUC-OpenPi

python run/scripts/run_pi_server.py \
  --policy-dir <checkpoint_step_dir> \
  --port 8000
```

Use the concrete checkpoint step directory, for example `<checkpoint_root>/<exp_name>/<step>`.

Client:

Run the rollout client after the OpenPi server is already serving requests.

```bash
conda activate activefly-pi
cd FLUC-OpenPi

python eval/run_rollout.py \
  -r <data_root>/ActiveFly_Test/<split> \
  -o <openpi_output_dir> \
  --server_port 8000 \
  --instruction_type instruction
```

Metrics:

Run the shared metric script after the OpenPi rollout.

```bash
python ../scripts/metric.py \
  <data_root>/ActiveFly_Test/<split> \
  <openpi_output_dir> \
  --metric all \
  --traj-len-thresh 5 \
  --output-dir <metric_output_dir>
```

The summary reports `nDTW`, `NE`, `SR`, and `OSR`.

Run `metric.py` only after a rollout client has finished writing trajectory outputs. It is the post-rollout evaluation step for OpenVLA and OpenPi.

### 3.3 End-to-end EP evaluation

**Full evaluation.** Configure the data paths and VLM API credentials in Section 2,
the scene executables, and the policy checkpoints (`OPENVLA_CHECKPOINT` and
`PI0_POLICY_DIR`). From the repository root, run:

```bash
bash run_pipeline.sh
```

By default this starts the policy servers and runs Beijing and CityPark in turn.
It selects every simulated EQA question in each scene and evaluates all six
pairings of Qwen, GPT and Gemini with OpenVLA and OpenPi.

To evaluate the **full question set with Qwen and OpenVLA only**, run:

```bash
bash run_pipeline.sh --model qwen --policy openvla
```

**Single-question check.** To reproduce one CityPark question with CityPark and
OpenVLA already running as in Section 3.2, use the policy server's actual port:

```bash
conda activate activefly-openvla
SKIP_CONDA=1 RUN_SIMULATOR=0 RUN_OPENVLA_SERVER=0 OPENVLA_SERVER_PORT=5008 \
  bash run_pipeline.sh --idx 76 --model qwen --policy openvla
```

Set the port to the one used by your policy server. For OpenPi, activate
`activefly-pi`, use `--policy pi`, and set `RUN_PI0_SERVER=0 PI0_SERVER_PORT=8000`.

For each EQA question, the planner sees the initial frame, that question's stem,
and the trajectory's OBP action options. The selected instruction drives a fresh
rollout, even when several questions share a starting pose. The VLM answers the
same EQA question using up to 16 frames from its own rollout. An incorrect action
choice or missed reference pose still proceeds to EQA.

EP success requires a correct OBP choice, FLUC oracle success and a correct EQA
answer. The released OBP label applies when the EQA stem matches the original
planning question and the labelled instruction is unique. Other completed
questions report their rollout and EQA results with null OBP and EP scores.
Malformed action menus are recorded as invalid inputs. The summary reports
selected, completed and scored counts; the EP rate uses the scored count.

Use `--idx` for one EQA question, `--data-id` for all questions linked to one
trajectory, or omit both for the configured simulated scenes. Without explicit
model and policy flags, EP uses the rollout pairs in `EQA_MODELS`. Each run
has its own directory under `eval/ep/`. Each `<data_id>/<question_id>/` directory
holds its `result.json` and, for completed questions, its rollout. The run-level
`summary.json` is written after processing the selected questions.

Use `bash run_pipeline.sh --stages` for independent component evaluations and
the existing result collection workflow. Standalone stage selections from the
web UI continue to use that mode.

### 3.4 Collect Evaluation Results

After EQA, OBP, and FLUC runs finish, use this script to summarize all saved outputs into `eval/` and build static debug pages under `eval/pipeline_web/`.

```bash
bash run_eval.sh
```

Then open the [Web UI](#5-web-ui) in debug mode to browse the results.

## 4. Training VLA Models

Use this section when you want to finetune the VLA models on the current ActiveFly training set instead of only running evaluation.

### 4.1 Train OpenVLA

Use the OpenVLA finetuning launcher from `FLUC-OpenVLA`.

The launcher reads trajectory folders directly from `ActiveFly_Train`, so no extra conversion step is required if your current data is already organized in that format.

```bash
conda activate activefly-openvla
cd FLUC-OpenVLA

sh run/vla-scripts/finetune_act.sh
```

To train from a specific pretrained checkpoint or data root, adjust the arguments inside `FLUC-OpenVLA/run/vla-scripts/finetune_act.sh` or call `FLUC-OpenVLA/run/vla-scripts/finetune_act.py` directly with your custom arguments.

### 4.2 Train OpenPi

OpenPi finetuning first converts `ActiveFly_Train` to LeRobot format, then starts training. By default both OpenVLA and OpenPi read `datasets/ActiveFly_Train`; set `ACTIVEFLY_FLUC_SIM_DIR` to override.

```bash
conda activate activefly-pi
cd FLUC-OpenPi

sh run/scripts/finetune_pi.sh
```

The launcher writes checkpoints under the data root configured in `FLUC-OpenPi/run/scripts/finetune_pi.sh` and computes missing norm stats before training starts.

To train from a specific pretrained checkpoint or data root, adjust the arguments inside `FLUC-OpenPi/run/scripts/finetune_pi.sh` or call `FLUC-OpenPi/run/scripts/finetune_pi.py` directly with your custom arguments.

## 5. Web UI

Interactive runner / debug UI:

Launch the interactive web UI for manual debugging and pipeline inspection. Make sure [dataset paths](#2-configuration) are set before launching.

```bash
python pipeline_web.py
```

Static debug view:

Use the static mode when you only need to inspect saved results without the interactive controls. Pages are served from `eval/pipeline_web/` (built by [run_eval.sh](#34-collect-evaluation-results)).

```bash
python pipeline_web.py --static
```

Debug mode reads trajectory frames from your configured dataset directories and model answers from `eval/api_results/`, then displays them as GIFs in the page. Run mode lets you execute the pipeline on one item from the browser; after the job finishes, open that item's detail page to view the updated videos and model answers.

## 5. Real-world Deployment
We provide the sample deployment code for real UAVs in [ActiveFly-Client-ROS](https://github.com/EmbodiedCity/ActiveFly-Bench.code/tree/main/ActiveFly-Client-ROS). Note that this repository only includes the VLA client code. The UAV flight control and path planning modules are not included. Please refer to [here](https://github.com/EmbodiedCity/ActiveFly-Bench.code/blob/main/ActiveFly-Client-ROS/README.md) for more details.

## 6. Acknowledgement
We sincerely thank the following projects for their excellent work: [UAV-Flow](https://prince687028.github.io/UAV-Flow/), [OpenVLA](https://github.com/openvla/openvla), [OpenPi](https://github.com/Physical-Intelligence/openpi).

## License
This project is released under the **CC BY 4.0** license.
