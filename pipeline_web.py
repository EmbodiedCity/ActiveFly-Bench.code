#!/usr/bin/env python3
"""
ActiveFly pipeline web: generate self-contained HTML pages with animated GIFs,
model responses, and correct answers.

Single item:
  python pipeline_web.py --idx 5
  python pipeline_web.py --qid 00-244
  python pipeline_web.py --dataset OBP --idx 0

Batch:
  python pipeline_web.py --static

Interactive runner:
  python pipeline_web.py
"""
import argparse, base64, functools, json, os, re, socket, subprocess, sys, threading, time, uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image

# ── paths ─────────────────────────────────────────────────────────────────────
ACTIVEFLY_ROOT = Path(__file__).resolve().parent


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if "#" in value:
            value = value.split("#", 1)[0].rstrip()
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


load_env_file(ACTIVEFLY_ROOT / ".env")
sys.path.insert(0, str(ACTIVEFLY_ROOT / "utils"))

from paths import FLUC_SIM_DIR, OPENVLA_DIR, PI0_DIR, QA_DIR, FLUC_REAL_DIR, ROLLOUT_DIR
from activefly_eval_data import default_prediction_store, item_id

ROOT = QA_DIR

RESULTS = ROOT / "results"

MODELS       = ["gemini", "gpt", "qwen"]
MODEL_LABELS = {"gemini": "Gemini", "gpt": "GPT", "qwen": "Qwen"}
VALID        = {"A", "B", "C", "D"}
PYTHON_BIN   = os.environ.get("PYTHON_BIN", sys.executable or "python3")
LOG_DIR      = ACTIVEFLY_ROOT / "test_logs" / "pipeline_web"
EVAL_DIR     = ACTIVEFLY_ROOT / "eval"
TRAJ_FLUC_JSONL = EVAL_DIR / "trajectory_fluc.jsonl"
STATIC_WEB_DIR = EVAL_DIR / "pipeline_web"
DEFAULT_HOST = os.environ.get("ACTIVEFLY_WEB_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("ACTIVEFLY_WEB_PORT", "8765"))
SECRET_API_FIELDS = {
    "openai_key",
    "gemini_key",
    "qwen_key",
}

VIEW_DEFS = [
    ("gt",      "Original (GT)",  ""),
    ("openvla", "OpenVLA",        "_openvla"),
    ("pi0",     "Pi0",            "_pi0"),
]

EQA_RUN_MODELS = os.environ.get(
    "PIPELINE_WEB_EQA_MODELS",
    "gemini gpt qwen gemini_pi0 gpt_pi0 qwen_pi0 gemini_openvla gpt_openvla qwen_openvla",
).split()
OBP_RUN_MODELS = os.environ.get("PIPELINE_WEB_OBP_MODELS", "gemini gpt qwen").split()

_REAL_MAP = None


def _clean_text(value):
    return str(value or "").strip()


def api_env_from_payload(payload):
    """The web runner does not collect API credentials.

    API keys and compatible base URLs are configured once through
    run_pipeline.sh, .env, or the process environment.
    """
    return {}


def sanitize_payload(payload):
    if not isinstance(payload, dict):
        return {}
    safe = dict(payload)
    api = safe.get("api")
    if isinstance(api, dict):
        safe_api = {}
        for key, value in api.items():
            if key in SECRET_API_FIELDS:
                safe_api[key] = "[provided]" if _clean_text(value) else ""
            else:
                safe_api[key] = value
        safe["api"] = safe_api
    return safe


def api_config_summary(payload):
    return "api=run_pipeline.sh/.env"


def payload_has_api_key(payload):
    return True


def _real_data_map():
    global _REAL_MAP
    if _REAL_MAP is None:
        _REAL_MAP = {p.parent.name: p.parent for p in FLUC_REAL_DIR.rglob("000000.png")} if FLUC_REAL_DIR.is_dir() else {}
    return _REAL_MAP


# ── frame helpers ─────────────────────────────────────────────────────────────

def _subsample(paths, n=16):
    if len(paths) <= n:
        return paths
    step = (len(paths) - 1) / (n - 1)
    return [paths[round(i * step)] for i in range(n)]


def get_frames(item, view, dataset="EQA"):
    """Return (display_paths, folder_str, total_count).
    OBP uses only the first frame; EQA subsamples up to 16."""
    cat, env, data_id = item["data_category"], item["environment"], item["data_id"]

    if view == "gt":
        if item.get("data_simorreal") == "real":
            folder = _real_data_map().get(data_id)
            if folder is None:
                return [], f"NOT FOUND: {FLUC_REAL_DIR}/**/{data_id}", 0
        elif cat == "action":
            action_dir = item["data_subcategory"]
            base = FLUC_SIM_DIR / "action" / "action_raw" / "action_all_mix_test" / env
            candidates = [base / action_dir / data_id, base / action_dir.replace(" ", "_") / data_id]
            folder = next((p for p in candidates if p.is_dir()), candidates[0])
        else:
            folder = FLUC_SIM_DIR / cat / "test" / env / data_id
        if not folder.exists():
            return [], str(folder), 0
        all_pngs = sorted(folder.glob("*.png"), key=lambda p: int(p.stem))
        sampled  = all_pngs[:1] if dataset == "OBP" else _subsample(all_pngs)
        return sampled, str(folder), len(all_pngs)

    if view == "pi0":
        direct = PI0_DIR / data_id / "frames"
        folder = direct if direct.is_dir() else None
        if folder is None:
            matches = sorted(p for p in PI0_DIR.glob(f"**/{data_id}/frames") if p.is_dir())
            folder = matches[0] if matches else None
        if folder is None:
            return [], f"NOT FOUND: {PI0_DIR / data_id / 'frames'}", 0
        all_pngs = sorted(folder.glob("*.png"), key=lambda p: int(p.stem))
        sampled = all_pngs[:1] if dataset == "OBP" else _subsample(all_pngs)
        return sampled, str(folder), len(all_pngs)

    if view == "openvla":
        direct = OPENVLA_DIR / data_id / "frames"
        folder = direct if direct.is_dir() else None
        if folder is None and cat == "action":
            action = item["data_subcategory"]
            candidates = [
                OPENVLA_DIR / cat / "cotraining_tpz" / env / action / data_id / "frames",
                OPENVLA_DIR / cat / "cotraining_tpz" / env / action.replace(" ", "_") / data_id / "frames",
            ]
            folder = next((p for p in candidates if p.is_dir()), None)
            search_root = OPENVLA_DIR / cat / "cotraining_tpz" / env
        elif folder is None:
            candidates = [OPENVLA_DIR / cat / "cotraining_tpz" / env / data_id / "frames"]
            folder = next((p for p in candidates if p.is_dir()), None)
            search_root = OPENVLA_DIR / cat / "cotraining_tpz" / env
        else:
            search_root = OPENVLA_DIR
        if folder is None:
            matches = sorted(p for p in search_root.glob(f"**/{data_id}/frames") if p.is_dir())
            folder = matches[0] if matches else None
        if folder is None:
            return [], "NOT FOUND", 0
        all_pngs = sorted(folder.glob("*.png"), key=lambda p: int(p.stem))
        sampled = all_pngs[:1] if dataset == "OBP" else _subsample(all_pngs)
        return sampled, str(folder), len(all_pngs)

    return [], "unknown view", 0


def make_gif_b64(paths, fps=4, max_size=480):
    if not paths:
        return None
    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        frames.append(img)
    buf = BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:],
        loop=0, duration=int(1000 / fps), optimize=False,
    )
    return base64.b64encode(buf.getvalue()).decode()


# ── result helpers ────────────────────────────────────────────────────────────

def load_results(dataset, model, view_suffix):
    p = RESULTS / f"{dataset}_{model}{view_suffix}.json"
    if not p.exists():
        return {}
    recs = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        qid = str(r.get("question_id", ""))
        if qid:
            recs[qid] = r
    return recs


# ── per-item HTML ─────────────────────────────────────────────────────────────

_ITEM_STYLE = """
body { font-family:-apple-system,"Segoe UI",sans-serif; margin:0; padding:20px;
       background:#f5f5f5; color:#222; }
h1   { font-size:1.15em; margin:0 0 6px; }
.meta { font-size:0.82em; color:#666; margin-bottom:14px; line-height:1.8; }
.qbox { background:#fff; border:1px solid #ddd; border-radius:8px;
        padding:14px 18px; margin-bottom:18px; line-height:1.65; white-space:pre-wrap; }
.badge { display:inline-block; background:#27ae60; color:#fff;
         border-radius:4px; padding:2px 9px; font-weight:bold; }
.cols { display:flex; gap:12px; margin-bottom:20px; align-items:flex-start; }
.col  { flex:1; min-width:0; background:#fff; border:1px solid #ddd;
        border-radius:8px; overflow:hidden; }
.col-head { padding:10px 14px; background:#f0f0f0; border-bottom:1px solid #ddd;
            font-weight:bold; font-size:0.95em; }
.col-body { padding:10px; }
.col-body img { width:100%; border-radius:4px; display:block; }
.no-frame { background:#eee; padding:24px; text-align:center; border-radius:4px;
            color:#888; font-size:0.85em; word-break:break-all; }
.model-row  { margin:8px 0; border:1px solid #e0e0e0; border-radius:6px; overflow:hidden; }
.model-head { padding:7px 12px; background:#fafafa; display:flex;
              justify-content:space-between; align-items:center;
              cursor:pointer; user-select:none; }
.model-head:hover { background:#f0f0f0; }
.model-body { display:none; padding:10px 12px; font-size:0.78em; white-space:pre-wrap;
              background:#fff; max-height:320px; overflow-y:auto;
              font-family:ui-monospace,monospace; border-top:1px solid #eee; }
.c { color:#27ae60; font-weight:bold; }
.w { color:#e74c3c; font-weight:bold; }
.n { color:#aaa; }
@media(max-width:800px){ .cols{ flex-direction:column; } }
"""

_ITEM_SCRIPT = "function toggle(id){var e=document.getElementById(id);e.style.display=e.style.display==='none'?'block':'none';}"


def build_item_html(item, dataset, qid, views_data, responses, correct):
    id_key = "question_id" if dataset == "EQA" else "obp_id"
    meta_parts = [
        ("category",    item.get("data_category", "")),
        ("subcategory", item.get("data_subcategory") or "—"),
        ("env",         item.get("environment", "")),
        ("data_id",     f'<code>{item.get("data_id","")}</code>'),
        ("q_type",      item.get("question_category", "")),
    ]
    meta_html = " &nbsp;|&nbsp; ".join(f"{k}: <b>{v}</b>" for k, v in meta_parts)
    question  = item.get("question", "").replace("&","&amp;").replace("<","&lt;")

    cols_html = ""
    for view_key, view_label, _, gif_b64, folder, total, n_sampled in views_data:
        frame_info = f"{n_sampled} shown / {total} total" if total else "0 frames"
        if gif_b64:
            media = f'<img src="data:image/gif;base64,{gif_b64}" alt="{view_label}">'
        else:
            media = f'<div class="no-frame">No frames found<br>{folder}</div>'

        model_rows = ""
        for m in MODELS:
            r  = responses[view_key][m]
            ch = r["choice"]
            if ch in VALID:
                css, icon = ("c","✓") if ch == correct else ("w","✗")
            else:
                css, icon = "n", "?"
            full = r["full"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            uid  = f"{view_key}_{m}"
            model_rows += f"""
            <div class="model-row">
              <div class="model-head" onclick="toggle('{uid}')">
                <span>{MODEL_LABELS[m]}</span>
                <span class="{css}">{icon} {ch}</span>
              </div>
              <div id="{uid}" class="model-body">{full}</div>
            </div>"""

        cols_html += f"""
        <div class="col">
          <div class="col-head">{view_label} <span style="font-weight:normal;color:#888;font-size:0.85em">({frame_info})</span></div>
          <div class="col-body">{media}<div style="margin-top:10px">{model_rows}</div></div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Debug {dataset} {qid}</title>
<style>{_ITEM_STYLE}</style>
<script>{_ITEM_SCRIPT}</script>
</head><body>
<h1>Debug: {dataset} &mdash; {qid}</h1>
<div class="meta">{meta_html}</div>
<div class="qbox"><b>Q:</b> {question}

Correct answer: <span class="badge">{correct}</span></div>
<div class="cols">{cols_html}</div>
<p style="font-size:0.78em;color:#aaa">Click a model row to expand/collapse reasoning.</p>
</body></html>"""


def process_item(item, dataset, out_path, fps, size, all_responses, skip_media=False):
    """Build and write one HTML file; return row dict for index."""
    id_key  = "question_id" if dataset == "EQA" else "obp_id"
    qid     = str(item.get(id_key))
    correct = item.get("answer", "?")

    views_data = []
    for view_key, view_label, view_suffix in VIEW_DEFS:
        paths, folder, total = get_frames(item, view_key, dataset)
        gif_b64 = None if skip_media else make_gif_b64(paths, fps=fps, max_size=size)
        views_data.append((view_key, view_label, view_suffix, gif_b64, folder, total, len(paths)))

    responses = {}
    for view_key, _, view_suffix in VIEW_DEFS:
        responses[view_key] = {}
        for m in MODELS:
            key = f"{dataset}_{m}{view_suffix}"
            r   = all_responses.get(key, {}).get(qid, {})
            responses[view_key][m] = {
                "choice": r.get("choice", "N/A"),
                "full":   r.get("full_response", "(result not found)"),
            }

    if out_path is not None:
        html = build_item_html(item, dataset, qid, views_data, responses, correct)
        out_path.write_text(html, encoding="utf-8")

    row = {
        "qid":     qid,
        "dataset": dataset,
        "cat":     item.get("data_category", ""),
        "subcat":  item.get("data_subcategory", ""),
        "env":     item.get("environment", ""),
        "data_id": item.get("data_id", ""),
        "qtype":   item.get("question_category", ""),
        "correct": correct,
        "file":    f"{dataset}/{qid.replace('/', '-')}.html",
        "frames":  {vk: n for vk, _, _, _, _, _, n in views_data},
        "choices": {
            f"{vk}_{m}": responses[vk][m]["choice"]
            for vk in ["gt","openvla","pi0"] for m in MODELS
        },
    }
    return qid, row


def _worker(args_tuple):
    try:
        return process_item(*args_tuple)
    except Exception as e:
        item, dataset, *_ = args_tuple
        id_key = "question_id" if dataset == "EQA" else "obp_id"
        qid = str(item.get(id_key, "?"))
        return qid, {"error": str(e), "qid": qid, "dataset": dataset}


# ── index HTML ────────────────────────────────────────────────────────────────

_INDEX_STYLE = """
*{box-sizing:border-box}
body{font-family:-apple-system,"Segoe UI",sans-serif;margin:0;padding:16px;background:#f5f5f5;font-size:13px}
h1{font-size:1.2em;margin-bottom:4px}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px;background:#fff;padding:10px;border-radius:6px;border:1px solid #ddd}
.controls label{font-size:.85em;color:#555}
.controls select,.controls input{padding:4px 7px;border:1px solid #ccc;border-radius:4px;font-size:.85em}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #ddd;border-radius:6px;overflow:hidden}
th{background:#eee;padding:7px 8px;text-align:left;font-size:.82em;border-bottom:2px solid #ccc;white-space:nowrap;cursor:pointer}
th:hover{background:#e0e0e0}
td{padding:5px 8px;border-bottom:1px solid #f0f0f0;vertical-align:middle}
tr:hover td{background:#f9f9f9}
a{color:#2563eb;text-decoration:none}
a:hover{text-decoration:underline}
.c{color:#27ae60;font-weight:bold}
.w{color:#e74c3c;font-weight:bold}
.n{color:#aaa}
.badge{display:inline-block;border-radius:3px;padding:1px 5px;font-size:.78em;font-weight:bold;color:#fff}
.b-action{background:#7c4dff}.b-occlusion{background:#0288d1}.b-orientation{background:#e65100}
.b-fuzzy{background:#6d4c41}.b-central_view{background:#388e3c}.b-other{background:#607d8b}
#count{font-size:.85em;color:#888;margin-bottom:6px}
"""

_INDEX_SCRIPT = r"""
const rows=JSON.parse(document.getElementById('row-data').textContent);
const tbody=document.getElementById('tbody');
const VALID=new Set(['A','B','C','D']);
const models=['gemini','gpt','qwen'];
function cls(c,ok){return!VALID.has(c)?'n':ok?'c':'w';}
function icon(c,ok){return!VALID.has(c)?'?':ok?'✓':'✗';}
function badge(cat){const m={action:'b-action',occlusion:'b-occlusion',orientation:'b-orientation',fuzzy:'b-fuzzy',central_view:'b-central_view'};return`<span class="badge ${m[cat]||'b-other'}">${cat}</span>`;}
let sc=null,sa=true;
function render(data){
  tbody.innerHTML='';
  data.forEach(r=>{
    if(r.error)return;
    let cells=`<td><a href="${r.file}">${r.qid}</a></td><td>${r.dataset}</td><td>${badge(r.cat)}</td><td>${r.subcat||'—'}</td><td>${r.env}</td><td>${r.qtype}</td><td><b>${r.correct}</b></td>`;
    ['gt','openvla','pi0'].forEach(v=>models.forEach(m=>{const ch=(r.choices||{})[`${v}_${m}`]||'?';const ok=VALID.has(ch)&&ch===r.correct;cells+=`<td class="${cls(ch,ok)}" title="${v}/${m}">${icon(ch,ok)}</td>`;}));
    tbody.innerHTML+=`<tr>${cells}</tr>`;
  });
  document.getElementById('count').textContent=`Showing ${data.length} / ${rows.length} items`;
}
function filtered(){
  const ds=document.getElementById('f-ds').value;
  const cat=document.getElementById('f-cat').value;
  const sv=document.getElementById('f-sc').value.toLowerCase().trim();
  const env=document.getElementById('f-env').value;
  const pat=document.getElementById('f-pat').value;
  return rows.filter(r=>{
    if(r.error)return false;
    if(ds&&r.dataset!==ds)return false;
    if(cat&&r.cat!==cat)return false;
    if(sv&&!r.subcat.toLowerCase().includes(sv))return false;
    if(env&&r.env!==env)return false;
    if(!pat)return true;
    const[vk,cond]=pat.split('=');
    const oks=models.map(m=>{const c=(r.choices||{})[`${vk}_${m}`]||'?';return VALID.has(c)&&c===r.correct;});
    if(cond==='all_correct')return oks.every(Boolean);
    if(cond==='any_correct')return oks.some(Boolean);
    if(cond==='all_wrong')return oks.every(x=>!x);
    if(cond==='any_wrong')return oks.some(x=>!x);
    return true;
  });
}
function apply(){render(filtered());}
['f-ds','f-cat','f-sc','f-env','f-pat'].forEach(id=>{
  document.getElementById(id).addEventListener('input',apply);
  document.getElementById(id).addEventListener('change',apply);
});
document.querySelectorAll('th[data-col]').forEach(th=>{
  th.addEventListener('click',()=>{
    const col=th.dataset.col;
    sc===col?sa=!sa:(sc=col,sa=true);
    const d=filtered();d.sort((a,b)=>sa?(a[col]||'').localeCompare(b[col]||'',undefined,{numeric:true}):(b[col]||'').localeCompare(a[col]||'',undefined,{numeric:true}));
    render(d);
  });
});
render(rows);
"""


def build_debug_index_html(all_rows, title="ActiveFly Debug Index", nav=False, heading=None):
    heading = heading or ("ActiveFly — Debug Index" if not nav else title)
    datasets = sorted({r["dataset"] for r in all_rows if "error" not in r})
    cats     = sorted({r["cat"]     for r in all_rows if "error" not in r})
    envs     = sorted({r["env"]     for r in all_rows if "error" not in r})

    def opts(vals, label):
        return f'<option value="">All {label}</option>' + "".join(
            f'<option value="{v}">{v}</option>' for v in vals)

    vm_heads = "".join(
        f'<th title="{v}/{m}">{v[:2]}/{MODEL_LABELS[m][:3]}</th>'
        for v in ["gt","openvla","pi0"] for m in MODELS)
    nav_html = '<div class="controls"><a href="/run">Pipeline Runner</a><a href="/debug">Debug Static</a></div>' if nav else ""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>{_INDEX_STYLE}</style>
</head><body>
<h1>{heading}</h1>
{nav_html}
<div class="controls">
  <div><label>Dataset<br><select id="f-ds">{opts(datasets,"datasets")}</select></label></div>
  <div><label>Category<br><select id="f-cat">{opts(cats,"categories")}</select></label></div>
  <div><label>Subcategory<br><input id="f-sc" placeholder="e.g. move right" style="width:130px"></label></div>
  <div><label>Env<br><select id="f-env">{opts(envs,"envs")}</select></label></div>
  <div><label>View filter<br>
    <select id="f-pat">
      <option value="">No filter</option>
      <option value="gt=all_wrong">GT all wrong</option>
      <option value="gt=all_correct">GT all correct</option>
      <option value="pi0=all_correct">Pi0 all correct</option>
      <option value="pi0=all_wrong">Pi0 all wrong</option>
      <option value="openvla=all_correct">OpenVLA all correct</option>
      <option value="openvla=all_wrong">OpenVLA all wrong</option>
    </select>
  </label></div>
</div>
<div id="count"></div>
<table>
<thead><tr>
  <th data-col="qid">QID</th><th data-col="dataset">DS</th>
  <th data-col="cat">Category</th><th data-col="subcat">Subcategory</th>
  <th data-col="env">Env</th><th data-col="qtype">Q-Type</th>
  <th data-col="correct">Ans</th>{vm_heads}
</tr></thead>
<tbody id="tbody"></tbody>
</table>
<script id="row-data" type="application/json">{json.dumps(all_rows, ensure_ascii=False)}</script>
<script>{_INDEX_SCRIPT}</script>
</body></html>"""


def build_index(all_rows, out_dir):
    html = build_debug_index_html(all_rows)
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ActiveFly web. Default starts the runner; add --static for the read-only debug web.")
    p.add_argument("--static", action="store_true", help="serve the read-only static debug web")
    p.add_argument("--dataset", choices=["EQA","OBP","both"], default="both", help=argparse.SUPPRESS)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--idx", type=int, help=argparse.SUPPRESS)
    g.add_argument("--qid", type=str, help=argparse.SUPPRESS)
    p.add_argument("--fps", type=float, default=float(os.environ.get("ACTIVEFLY_WEB_FPS", "4")), help=argparse.SUPPRESS)
    p.add_argument("--size", type=int, default=int(os.environ.get("ACTIVEFLY_WEB_SIZE", "480")), help=argparse.SUPPRESS)
    p.add_argument("--out", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--workers", type=int, default=int(os.environ.get("ACTIVEFLY_WEB_WORKERS", "32")), help=argparse.SUPPRESS)
    p.add_argument("--filter", type=str, default=os.environ.get("ACTIVEFLY_WEB_FILTER"), help=argparse.SUPPRESS)
    p.add_argument("--skip-media", action="store_true", default=os.environ.get("ACTIVEFLY_WEB_SKIP_MEDIA", "0") == "1", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.host = DEFAULT_HOST
    args.port = DEFAULT_PORT
    args.dry_run = os.environ.get("ACTIVEFLY_WEB_DRY_RUN", "0") == "1"
    return args


def _load_all_responses(datasets):
    store = default_prediction_store()
    all_responses = {}
    for ds in datasets:
        data_path = ROOT / f"{ds}.json"
        data = json.loads(data_path.read_text(encoding="utf-8")) if data_path.exists() else []
        for m in MODELS:
            for view_key, _, view_suffix in VIEW_DEFS:
                key = f"{ds}_{m}{view_suffix}"
                recs = {}
                view_name = "original" if view_key == "gt" else view_key
                for item in data:
                    qid = item_id(ds, item)
                    pred = store.get(ds, qid, view_name, m)
                    if pred.exists:
                        recs[qid] = {"choice": pred.choice, "full_response": pred.full_response}
                all_responses[key] = recs
    return all_responses


def load_dataset(dataset):
    path = ROOT / f"{dataset}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _qid_for_item(dataset, item):
    return item_id(dataset, item)


def find_item_by_qid(dataset, qid):
    for idx, item in enumerate(load_dataset(dataset)):
        if str(_qid_for_item(dataset, item)) == str(qid):
            return idx, item
    return None, None


def natural_qid_parts(qid):
    return [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', str(qid))]


def row_sort_key(row):
    return (row.get("dataset", ""), natural_qid_parts(row.get("qid", "")))


def indices_for_data_id(dataset, data_id):
    return [
        (idx, item)
        for idx, item in enumerate(load_dataset(dataset))
        if str(item.get("data_id", "")) == str(data_id)
    ]


def run_single(args):
    datasets = ["EQA","OBP"] if args.dataset == "both" else [args.dataset]
    # find item
    item = dataset = None
    for ds in datasets:
        data   = json.loads((ROOT / f"{ds}.json").read_text())
        id_key = "question_id" if ds == "EQA" else "obp_id"
        if args.idx is not None:
            if args.idx < len(data):
                item, dataset = data[args.idx], ds
                break
        else:
            found = next((x for x in data if str(x.get(id_key)) == args.qid), None)
            if found:
                item, dataset = found, ds
                break
    if item is None:
        sys.exit(f"Item not found (idx={args.idx} qid={args.qid} dataset={args.dataset})")

    id_key  = "question_id" if dataset == "EQA" else "obp_id"
    qid     = str(item.get(id_key))
    print(f"  {dataset}  qid={qid}  cat={item.get('data_category')}  subcat={item.get('data_subcategory','')}  ans={item.get('answer')}")

    all_responses = _load_all_responses([dataset])
    out_path = Path(args.out) if args.out else ACTIVEFLY_ROOT / "eval" / f"pipeline_{dataset}_{qid.replace('/','-')}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _, row = process_item(item, dataset, out_path, args.fps, args.size, all_responses, args.skip_media)
    print(f"  frames: {row['frames']}")
    print(f"  → {out_path}")


def run_batch(args):
    datasets = ["EQA","OBP"] if args.dataset == "both" else [args.dataset]
    out_dir  = STATIC_WEB_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading result files...")
    all_responses = _load_all_responses(datasets)
    print(f"  {len(all_responses)} result files loaded")

    filter_key = filter_val = None
    if args.filter:
        filter_key, _, filter_val = args.filter.partition("=")
        filter_val = filter_val.lower().replace(" ", "_")
        print(f"  filter: {filter_key}={filter_val}")

    all_tasks = []
    for ds in datasets:
        data   = json.loads((ROOT / f"{ds}.json").read_text())
        id_key = "question_id" if ds == "EQA" else "obp_id"
        ds_out = out_dir / ds
        ds_out.mkdir(exist_ok=True)
        for item in data:
            if filter_key == "cat" and item.get("data_category","") != filter_val:
                continue
            if filter_key == "subcat":
                sc = item.get("data_subcategory","").lower().replace(" ","_")
                if sc != filter_val:
                    continue
            qid      = str(item.get(id_key))
            out_path = ds_out / f"{qid.replace('/','-')}.html"
            all_tasks.append((item, ds, out_path, args.fps, args.size, all_responses, args.skip_media))

    total = len(all_tasks)
    print(f"Generating {total} HTML pages ({args.workers} workers)...")

    all_rows = []
    done = errors = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_worker, t): t for t in all_tasks}
        for fut in as_completed(futs):
            qid, row = fut.result()
            all_rows.append(row)
            if "error" in row:
                errors += 1
                print(f"  ERROR {qid}: {row['error']}")
            else:
                done += 1
            if (done + errors) % 100 == 0:
                print(f"  {done+errors}/{total}  ({errors} errors)")

    print(f"Done: {done} OK, {errors} errors")
    all_rows.sort(key=row_sort_key)

    print("Building index.html...")
    build_index(all_rows, out_dir)

    print(f"\n  index  : {out_dir}/index.html")


def make_server(host, port, handler):
    last_error = None
    for candidate in range(port, port + 50):
        try:
            return ThreadingHTTPServer((host, candidate), handler), candidate
        except OSError as exc:
            last_error = exc
    raise RuntimeError(f"Could not bind a port from {port} to {port + 49}: {last_error}")


def display_url(host, port):
    if host in {"0.0.0.0", "::"}:
        return f"http://127.0.0.1:{port}/"
    return f"http://{host}:{port}/"


def run_static_server(args):
    if args.idx is not None or args.qid is not None:
        sys.exit("--static serves the batch debug web; omit item selectors")
    run_batch(args)
    if os.environ.get("ACTIVEFLY_STATIC_BUILD_ONLY") == "1":
        return
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(STATIC_WEB_DIR))
    server, port = make_server(args.host, args.port, handler)
    print(f"ActiveFly static debug web: {display_url(args.host, port)}")
    print(f"serving: {STATIC_WEB_DIR}")
    server.serve_forever()


# ── Pipeline demo server ──────────────────────────────────────────────────────

JOBS = {}
JOBS_LOCK = threading.Lock()


def append_job(job_id, text):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job["log"].append(text)


def finish_job(job_id, status, code=0):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            job["status"] = status
            job["returncode"] = code
            job["finished_at"] = time.time()


def run_cmd(job_id, cmd, env=None, cwd=ACTIVEFLY_ROOT, dry_run=False):
    printable = " ".join(str(x) for x in cmd)
    append_job(job_id, f"$ {printable}")
    if dry_run:
        append_job(job_id, "[dry-run] command not executed")
        return 0
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.stdout:
        append_job(job_id, proc.stdout.rstrip())
    if proc.returncode:
        append_job(job_id, f"[exit {proc.returncode}]")
    return proc.returncode


def collect_eval(job_id, dry_run=False, env=None, data_id=None):
    cmd = [PYTHON_BIN, ACTIVEFLY_ROOT / "utils" / "collect_eval.py", "--out", EVAL_DIR]
    if data_id:
        cmd.extend(["--data-id", data_id])
    return run_cmd(job_id, cmd, env=env, dry_run=dry_run)


def pipeline_stage_env(stage, api_env=None, data_id="", idx=None, idx_dataset="EQA"):
    env = os.environ.copy()
    if api_env:
        env.update(api_env)
    env["EVAL_DIR"] = str(EVAL_DIR)
    if idx is not None and str(idx) != "":
        env["ACTIVEFLY_IDX"] = str(idx)
        env["ACTIVEFLY_IDX_DATASET"] = str(idx_dataset or "EQA")
    if data_id:
        env["ACTIVEFLY_DATA_ID"] = str(data_id)
    if stage == "pipeline":
        return env
    stage_overrides = {
        "RUN_SIMULATOR": "0",
        "RUN_OPENVLA_SERVER": "0",
        "RUN_PI0_SERVER": "0",
        "RUN_OPENVLA_ROLLOUT": "0",
        "RUN_PI0_ROLLOUT": "0",
        "RUN_COLLECT_EVAL": "1",
    }
    if stage == "OBP":
        stage_overrides.update({"RUN_EQA": "0", "RUN_OBP": "1", "RUN_FLUC": "0"})
    elif stage == "EQA":
        stage_overrides.update({"RUN_EQA": "1", "RUN_OBP": "0", "RUN_FLUC": "0"})
    elif stage == "FLUC":
        stage_overrides.update({"RUN_EQA": "0", "RUN_OBP": "0", "RUN_FLUC": "1"})
    else:
        raise ValueError(f"unknown stage: {stage}")
    env.update(stage_overrides)
    return env


def run_pipeline_script(job_id, stage, dry_run=False, api_env=None, data_id="", idx=None, idx_dataset="EQA"):
    try:
        env = pipeline_stage_env(stage, api_env=api_env, data_id=data_id, idx=idx, idx_dataset=idx_dataset)
    except ValueError as exc:
        append_job(job_id, str(exc))
        return 2
    return run_cmd(job_id, ["bash", ACTIVEFLY_ROOT / "run_pipeline.sh"], env=env, dry_run=dry_run)


def run_global_stage(job_id, stage, dry_run=False, api_env=None):
    if stage not in {"pipeline", "OBP", "EQA", "FLUC"}:
        append_job(job_id, f"unknown stage: {stage}")
        return 2
    return run_pipeline_script(job_id, stage, dry_run=dry_run, api_env=api_env)


def run_answer_stage(job_id, dataset, data_id, models, dry_run=False, env=None):
    matches = indices_for_data_id(dataset, data_id)
    if not matches:
        append_job(job_id, f"No {dataset} items found for data_id={data_id}")
        return 0
    code = 0
    for idx, item in matches:
        qid = _qid_for_item(dataset, item)
        append_job(job_id, f"[{dataset}] data_id={data_id} id={qid} idx={idx}")
        for model in models:
            script = ROOT / f"test_{model}.py"
            if not script.exists():
                append_job(job_id, f"skip missing script: {script.name}")
                continue
            rc = run_cmd(job_id, [PYTHON_BIN, script, "--dataset", dataset, "--idx", str(idx)], env=env, dry_run=dry_run)
            code = code or rc
            if rc:
                return code
    return code


def run_fluc_for_trajectory(job_id, data_id, dry_run=False):
    matches = indices_for_data_id("EQA", data_id)
    if not matches:
        append_job(job_id, f"No EQA trajectory found for data_id={data_id}; FLUC uses EQA trajectory metadata")
        return 0
    idx, item = matches[0]
    append_job(job_id, f"[FLUC] data_id={data_id} using EQA idx={idx}")
    if dry_run:
        append_job(job_id, "[dry-run] would call metric_qa.fluc_metrics for openvla and pi0")
        return 0
    sys.path.insert(0, str(ROOT))
    try:
        import metric_qa
        metrics = {
            "openvla": metric_qa.fluc_metrics(item, "_openvla"),
            "pi0": metric_qa.fluc_metrics(item, "_pi0"),
        }
    except Exception as exc:
        append_job(job_id, f"FLUC failed: {exc}")
        return 1
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": "FLUC",
        "data_id": data_id,
        "eqa_index": idx,
        "metrics": metrics,
        "created_at": time.time(),
    }
    with TRAJ_FLUC_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    append_job(job_id, json.dumps(record, ensure_ascii=False))
    return 0


def run_trajectory_stage(job_id, stage, data_id="", idx=None, idx_dataset="EQA", dry_run=False, api_env=None):
    if (idx is None or str(idx) == "") and not data_id:
        append_job(job_id, "missing trajectory idx/data_id")
        return 2
    if stage not in {"pipeline", "OBP", "EQA", "FLUC"}:
        append_job(job_id, f"unknown stage: {stage}")
        return 2
    return run_pipeline_script(
        job_id,
        stage,
        dry_run=dry_run,
        api_env=api_env,
        data_id=data_id,
        idx=idx,
        idx_dataset=idx_dataset,
    )


def run_job(job_id, payload, dry_run=False):
    try:
        scope = payload.get("scope", "trajectory")
        stage = payload.get("stage", "pipeline")
        append_job(job_id, f"scope={scope} stage={stage}")
        api_env = api_env_from_payload(payload)
        append_job(job_id, api_config_summary(payload))
        if scope == "all":
            code = run_global_stage(job_id, stage, dry_run=dry_run, api_env=api_env)
        else:
            data_id = payload.get("data_id", "")
            idx = payload.get("idx")
            idx_dataset = payload.get("idx_dataset") or payload.get("dataset") or "EQA"
            code = run_trajectory_stage(
                job_id,
                stage,
                data_id=data_id,
                idx=idx,
                idx_dataset=idx_dataset,
                dry_run=dry_run,
                api_env=api_env,
            )
        finish_job(job_id, "failed" if code else "done", code)
    except Exception as exc:
        append_job(job_id, f"ERROR: {exc}")
        finish_job(job_id, "failed", 1)


def create_job(payload, dry_run=False):
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "running",
            "returncode": None,
            "created_at": time.time(),
            "finished_at": None,
            "payload": sanitize_payload(payload),
            "log": [],
        }
    thread = threading.Thread(target=run_job, args=(job_id, payload, dry_run), daemon=True)
    thread.start()
    return job_id


def json_response(handler, payload, status=200):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def build_server_index(rows, dry_run=False):
    datasets = sorted({r["dataset"] for r in rows if "error" not in r})
    cats = sorted({r["cat"] for r in rows if "error" not in r})
    envs = sorted({r["env"] for r in rows if "error" not in r})

    def opts(vals, label):
        return f'<option value="">All {label}</option>' + "".join(
            f'<option value="{v}">{v}</option>' for v in vals)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>ActiveFly Pipeline Demo</title>
<style>{_INDEX_STYLE}
.topbar{{display:flex;gap:10px;align-items:center;margin:10px 0 12px;flex-wrap:wrap}}
button{{border:1px solid #bbb;background:#fff;border-radius:5px;padding:4px 8px;font-size:12px;cursor:pointer}}
button:hover{{background:#f0f0f0}}
button:disabled{{opacity:.45;cursor:not-allowed}}
.run-all{{display:inline-flex;align-items:center;gap:8px;height:32px;border-color:#222;background:#222;color:#fff;font-weight:700}}
.run-all:hover{{background:#333}}
.flow-text{{font-weight:500;color:#d7d7d7}}
.play-triangle{{display:inline-block;width:0;height:0;border-top:5px solid transparent;border-bottom:5px solid transparent;border-left:8px solid currentColor}}
.metric-note{{font-size:12px;color:#555}}
.mode-nav{{display:flex;gap:8px;align-items:center;margin:4px 0 12px}}
.mode-nav a{{border:1px solid #ddd;background:#fff;border-radius:5px;padding:5px 9px;color:#222;text-decoration:none}}
.mode-nav a.active{{background:#222;color:#fff;border-color:#222}}
.pipeline-flow{{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}}
.run-icon{{width:28px;height:28px;padding:0;display:inline-flex;align-items:center;justify-content:center;border-color:#222;color:#222}}
.stage-btn{{height:28px;padding:3px 8px;border-color:#cfcfcf;background:#fafafa;font-weight:700}}
.stage-obp{{border-color:#7c4dff;color:#4b2db8}}
.stage-eqa{{border-color:#0288d1;color:#015f91}}
.stage-fluc{{border-color:#388e3c;color:#236425}}
.arrow{{color:#777;font-weight:700}}
#jobs{{background:#111;color:#eee;border-radius:6px;padding:10px;min-height:120px;max-height:280px;overflow:auto;font-family:ui-monospace,monospace;font-size:12px;white-space:pre-wrap}}
.pill{{background:#eee;border-radius:999px;padding:3px 8px;color:#555}}
</style></head><body>
<h1>ActiveFly Pipeline Demo</h1>
<div class="mode-nav"><a href="/debug">Debug Static</a><a class="active" href="/run">Pipeline Runner</a></div>
<div class="topbar">
  <button class="run-all" onclick="runAll('pipeline')" title="Run all trajectories: OBP to EQA to FLUC"><span class="play-triangle"></span><span>Run All</span><span class="flow-text">OBP &rarr; EQA &rarr; FLUC</span></button>
  <span class="pill">{'dry run' if dry_run else 'execute mode'}</span>
  <span class="metric-note" title="Metric code normalization follows Air-EQA/metric_qa.py and is kept centralized for future code mappings.">metric rules follow metric_qa.py</span>
</div>
<div class="controls">
  <div><label>Dataset<br><select id="f-ds">{opts(datasets,"datasets")}</select></label></div>
  <div><label>Category<br><select id="f-cat">{opts(cats,"categories")}</select></label></div>
  <div><label>Subcategory<br><input id="f-sc" placeholder="e.g. move right" style="width:130px"></label></div>
  <div><label>Env<br><select id="f-env">{opts(envs,"envs")}</select></label></div>
  <div><label>View filter<br><select id="f-pat"><option value="">No filter</option><option value="gt=all_wrong">GT all wrong</option><option value="gt=all_correct">GT all correct</option><option value="pi0=all_correct">Pi0 all correct</option><option value="pi0=all_wrong">Pi0 all wrong</option><option value="openvla=all_correct">OpenVLA all correct</option><option value="openvla=all_wrong">OpenVLA all wrong</option></select></label></div>
</div>
<div id="count"></div>
<table><thead><tr>
  <th data-col="qid">QID</th><th data-col="dataset">DS</th><th data-col="cat">Category</th>
  <th data-col="subcat">Subcategory</th><th data-col="env">Env</th><th data-col="qtype">Q-Type</th>
  <th data-col="correct">Ans</th><th>Pipeline</th>
</tr></thead><tbody id="tbody"></tbody></table>
<h2 style="font-size:1em;margin-top:16px">Run Log</h2>
<div id="jobs"></div>
<script id="row-data" type="application/json">{json.dumps(rows, ensure_ascii=False)}</script>
<script>
const rows=JSON.parse(document.getElementById('row-data').textContent);
const tbody=document.getElementById('tbody');
const VALID=new Set(['A','B','C','D']);
const models=['gemini','gpt','qwen'];
function badge(cat){{const m={{action:'b-action',occlusion:'b-occlusion',orientation:'b-orientation',fuzzy:'b-fuzzy',central_view:'b-central_view'}};return`<span class="badge ${{m[cat]||'b-other'}}">${{cat}}</span>`;}}
function stageButton(stage,label,klass,payload){{return `<button class="stage-btn ${{klass}}" title="${{label}}" onclick='runTrajectory("${{stage}}",${{payload}})'>${{stage}}</button>`;}}
function runButtons(r){{const payload=JSON.stringify({{idx:r.idx,dataset:r.dataset,data_id:r.data_id||''}});return `<div class="pipeline-flow"><button class="run-icon" title="Run pipeline: OBP to EQA to FLUC" aria-label="Run pipeline" onclick='runTrajectory("pipeline",${{payload}})'><span class="play-triangle"></span></button>${{stageButton('OBP','OBP ACC stage','stage-obp',payload)}}<span class="arrow">&rarr;</span>${{stageButton('EQA','EQA ACC stage','stage-eqa',payload)}}<span class="arrow">&rarr;</span>${{stageButton('FLUC','FLUC SR/OSR/nDTW stage','stage-fluc',payload)}}</div>`;}}
let sc=null,sa=true;
function render(data){{tbody.innerHTML='';data.forEach(r=>{{if(r.error)return;let cells=`<td><a href="${{r.file}}">${{r.qid}}</a></td><td>${{r.dataset}}</td><td>${{badge(r.cat)}}</td><td>${{r.subcat||'—'}}</td><td>${{r.env}}</td><td>${{r.qtype}}</td><td><b>${{r.correct}}</b></td><td>${{runButtons(r)}}</td>`;tbody.innerHTML+=`<tr>${{cells}}</tr>`;}});document.getElementById('count').textContent=`Showing ${{data.length}} / ${{rows.length}} items`;if(typeof updateRunAvailability==='function')updateRunAvailability();}}
function filtered(){{const ds=document.getElementById('f-ds').value;const cat=document.getElementById('f-cat').value;const sv=document.getElementById('f-sc').value.toLowerCase().trim();const env=document.getElementById('f-env').value;const pat=document.getElementById('f-pat').value;return rows.filter(r=>{{if(r.error)return false;if(ds&&r.dataset!==ds)return false;if(cat&&r.cat!==cat)return false;if(sv&&!r.subcat.toLowerCase().includes(sv))return false;if(env&&r.env!==env)return false;if(!pat)return true;const[vk,cond]=pat.split('=');const oks=models.map(m=>{{const c=(r.choices||{{}})[`${{vk}}_${{m}}`]||'';return VALID.has(c)&&c===r.correct;}});if(cond==='all_correct')return oks.every(Boolean);if(cond==='any_correct')return oks.some(Boolean);if(cond==='all_wrong')return oks.every(x=>!x);if(cond==='any_wrong')return oks.some(x=>!x);return true;}});}}
function apply(){{render(filtered());}}
['f-ds','f-cat','f-sc','f-env','f-pat'].forEach(id=>{{document.getElementById(id).addEventListener('input',apply);document.getElementById(id).addEventListener('change',apply);}});
document.querySelectorAll('th[data-col]').forEach(th=>{{th.addEventListener('click',()=>{{const col=th.dataset.col;sc===col?sa=!sa:(sc=col,sa=true);const d=filtered();d.sort((a,b)=>sa?(a[col]||'').localeCompare(b[col]||'',undefined,{{numeric:true}}):(b[col]||'').localeCompare(a[col]||'',undefined,{{numeric:true}}));render(d);}});}});
async function postRun(payload){{const res=await fetch('/api/run',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}});const data=await res.json();if(!res.ok){{document.getElementById('jobs').textContent=data.error||'request failed';return;}}pollJob(data.job_id);}}
function runAll(stage){{postRun({{scope:'all',stage}});}}
function runTrajectory(stage,target){{postRun({{scope:'trajectory',stage,idx:target.idx,idx_dataset:target.dataset,dataset:target.dataset,data_id:target.data_id||''}});}}
async function pollJob(id){{const box=document.getElementById('jobs');for(;;){{const res=await fetch('/api/jobs/'+id);const job=await res.json();box.textContent=`job ${{id}} [${{job.status}}]\\n`+(job.log||[]).join('\\n');box.scrollTop=box.scrollHeight;if(job.status!=='running')break;await new Promise(r=>setTimeout(r,1000));}}}}
render(rows);
</script></body></html>"""


def build_rows_for_server(args):
    datasets = ["EQA", "OBP"] if args.dataset == "both" else [args.dataset]
    all_responses = _load_all_responses(datasets)
    rows = []
    for ds in datasets:
        for idx, item in enumerate(load_dataset(ds)):
            qid = str(_qid_for_item(ds, item))
            _, row = process_item(
                item,
                ds,
                None,
                args.fps,
                args.size,
                all_responses,
                skip_media=True,
            )
            row["idx"] = idx
            row["file"] = f"/debug/{ds}/{qid.replace('/','-')}.html"
            rows.append(row)
    rows.sort(key=row_sort_key)
    return rows


def run_server(args):
    rows = build_rows_for_server(args)
    dry_run = args.dry_run

    def item_body(dataset, qid):
        idx, item = find_item_by_qid(dataset, qid)
        if item is None:
            return None
        all_responses = _load_all_responses([dataset])
        views_data = []
        for view_key, view_label, view_suffix in VIEW_DEFS:
            paths, folder, total = get_frames(item, view_key, dataset)
            gif_b64 = make_gif_b64(paths, fps=args.fps, max_size=args.size)
            views_data.append((view_key, view_label, view_suffix, gif_b64, folder, total, len(paths)))
        responses = {}
        for view_key, _, view_suffix in VIEW_DEFS:
            responses[view_key] = {}
            for model in MODELS:
                key = f"{dataset}_{model}{view_suffix}"
                rec = all_responses.get(key, {}).get(qid, {})
                responses[view_key][model] = {
                    "choice": rec.get("choice", "N/A"),
                    "full": rec.get("full_response", "(result not found)"),
                }
        return build_item_html(item, dataset, qid, views_data, responses, item.get("answer", "?")).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *params):
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % params))

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/run"}:
                body = build_server_index(rows, dry_run=dry_run).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/debug":
                body = build_debug_index_html(rows, title="ActiveFly Debug Static", nav=True).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            m = re.fullmatch(r"/debug/(EQA|OBP)/(.+)\.html", parsed.path)
            if m:
                dataset = m.group(1)
                qid = unquote(m.group(2))
                body = item_body(dataset, qid)
                if body is None:
                    self.send_error(404, "item not found")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/item":
                qs = parse_qs(parsed.query)
                dataset = qs.get("dataset", ["EQA"])[0]
                qid = qs.get("qid", [""])[0]
                body = item_body(dataset, qid)
                if body is None:
                    self.send_error(404, "item not found")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                with JOBS_LOCK:
                    job = JOBS.get(job_id)
                if job is None:
                    json_response(self, {"error": "job not found"}, status=404)
                else:
                    json_response(self, job)
                return
            self.send_error(404)

        def do_POST(self):
            if self.path != "/api/run":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if payload.get("stage") not in {"pipeline", "OBP", "EQA", "FLUC"}:
                json_response(self, {"error": "invalid stage"}, status=400)
                return
            if payload.get("scope", "trajectory") not in {"trajectory", "all"}:
                json_response(self, {"error": "invalid scope"}, status=400)
                return
            job_id = create_job(payload, dry_run=dry_run)
            json_response(self, {"job_id": job_id})

    server, port = make_server(args.host, args.port, Handler)
    print(f"ActiveFly pipeline runner: {display_url(args.host, port)}")
    print(f"debug view: {display_url(args.host, port)}debug")
    print(f"runner view: {display_url(args.host, port)}run")
    print("mode:", "dry-run" if dry_run else "execute")
    server.serve_forever()


def main():
    args = parse_args()
    if args.static:
        run_static_server(args)
        return
    if args.idx is not None or args.qid is not None:
        run_single(args)
    else:
        run_server(args)


if __name__ == "__main__":
    main()
