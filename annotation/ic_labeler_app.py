#!/usr/bin/env python3
"""Minimal browser labeler for MELODIC ICA outputs.

Input session layout:
  <ica-root>/<subject>/<session>/filtered_func_data.ica/melodic_IC.nii.gz

Completed output layout:
  <label-root>/<subject>/<session>/hand_labels_noise.txt

The exported label format is compatible with the project's existing FIX-style
``hand_labels_noise.txt`` files: noise=True and signal=False.
"""
from __future__ import annotations

import argparse
import io
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


class Labeler:
    def __init__(self, ica_root: Path, label_root: Path) -> None:
        self.ica_root = ica_root.resolve()
        self.label_root = label_root.resolve()
        self.lock = threading.Lock()
        self.sessions = self._index_sessions()

    def _index_sessions(self) -> dict[str, Path]:
        found: dict[str, Path] = {}
        for path in sorted(self.ica_root.glob("*/*/filtered_func_data.ica/melodic_IC.nii.gz")):
            key = str(path.parent.parent.relative_to(self.ica_root)).replace("\\", "/")
            found[key] = path
        return found

    def _state_path(self, key: str) -> Path:
        return self.label_root / key / "annotation_state.json"

    def _label_path(self, key: str) -> Path:
        return self.label_root / key / "hand_labels_noise.txt"

    def n_components(self, key: str) -> int:
        if key not in self.sessions:
            raise KeyError("unknown session")
        shape = nib.load(str(self.sessions[key])).shape
        if len(shape) != 4 or shape[3] < 1:
            raise ValueError("melodic_IC.nii.gz is not a 4D IC image")
        return int(shape[3])

    def state(self, key: str) -> dict:
        count = self.n_components(key)
        state_file = self._state_path(key)
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                state = {}
        else:
            state = {}
        labels = {str(k): v for k, v in state.get("labels", {}).items() if v in {"signal", "noise"}}
        return {"n_components": count, "labels": labels}

    def set_label(self, key: str, ic: int, label: str) -> dict:
        if label not in {"signal", "noise"}:
            raise ValueError("label must be signal or noise")
        count = self.n_components(key)
        if not 1 <= ic <= count:
            raise ValueError("IC index out of range")
        with self.lock:
            state = self.state(key)
            state["labels"][str(ic)] = label
            path = self._state_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return state

    def export(self, key: str) -> Path:
        with self.lock:
            state = self.state(key)
            count, labels = state["n_components"], state["labels"]
            missing = [str(i) for i in range(1, count + 1) if str(i) not in labels]
            if missing:
                raise ValueError(f"cannot export: {len(missing)} ICs are still unlabeled")
            noise = [i for i in range(1, count + 1) if labels[str(i)] == "noise"]
            lines = ["./filtered_func_data.ica"]
            for i in range(1, count + 1):
                if labels[str(i)] == "noise":
                    lines.append(f"{i}, Unclassified Noise, True")
                else:
                    lines.append(f"{i}, Signal, False")
            lines.append("[" + ", ".join(map(str, noise)) + "]")
            output = self._label_path(key)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return output

    def preview(self, key: str, ic: int) -> bytes:
        count = self.n_components(key)
        if not 1 <= ic <= count:
            raise ValueError("IC index out of range")
        data = np.asanyarray(nib.load(str(self.sessions[key])).dataobj[..., ic - 1], dtype=np.float32)
        finite = data[np.isfinite(data)]
        vmax = float(np.percentile(np.abs(finite), 99.5)) if finite.size else 1.0
        vmax = max(vmax, 1e-6)
        peak = np.unravel_index(np.nanargmax(np.abs(data)), data.shape)
        views = [data[peak[0], :, :].T, data[:, peak[1], :].T, data[:, :, peak[2]].T]
        names = ["sagittal", "coronal", "axial"]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), constrained_layout=True)
        for ax, image, name in zip(axes, views, names):
            ax.imshow(np.rot90(image), cmap="seismic", vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_title(name)
            ax.axis("off")
        fig.suptitle(f"{key} — IC {ic}")
        stream = io.BytesIO()
        fig.savefig(stream, format="png", dpi=130)
        plt.close(fig)
        return stream.getvalue()


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>IC labeler</title>
<style>body{font:16px system-ui;margin:26px auto;max-width:1050px;color:#202124}select,button{font:inherit;padding:8px;margin:4px}button{cursor:pointer}.signal{background:#e8f4ea}.noise{background:#fbe9e7}#preview{display:block;width:100%;max-width:1000px;border:1px solid #ddd;margin-top:12px}#status{color:#555;margin:10px 0}</style></head><body>
<h2>ICA component labeling</h2><select id="session"></select><button onclick="loadSession()">Load</button><span id="status"></span>
<div><button onclick="move(-1)">Previous</button><strong id="ic"></strong><button onclick="move(1)">Next</button><button class="signal" onclick="label('signal')">Signal</button><button class="noise" onclick="label('noise')">Noise</button><button onclick="exportLabels()">Export FIX-style labels</button></div>
<img id="preview" alt="IC spatial map preview"><script>
let key='',n=0,i=1,labels={}; const q=s=>document.querySelector(s);
async function api(url,opt){const r=await fetch(url,opt); const x=await r.json(); if(!r.ok)throw Error(x.error); return x}
async function init(){const x=await api('/api/sessions');q('#session').innerHTML=x.sessions.map(k=>`<option>${k}</option>`).join('');loadSession()}
async function loadSession(){key=q('#session').value;const x=await api('/api/state?key='+encodeURIComponent(key));n=x.n_components;labels=x.labels;i=1;show()}
function show(){q('#ic').textContent=`IC ${i} / ${n}`;q('#status').textContent=`labeled ${Object.keys(labels).length}/${n}`+(labels[String(i)]?` — current: ${labels[String(i)]}`:'');q('#preview').src='/preview?key='+encodeURIComponent(key)+'&ic='+i+'&t='+Date.now()}
function move(d){i=Math.max(1,Math.min(n,i+d));show()}
async function label(v){try{const x=await api('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:key,ic:i,label:v})});labels=x.labels;if(i<n)i++;show()}catch(e){alert(e.message)}}
async function exportLabels(){try{const x=await api('/api/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:key})});alert('Saved: '+x.path)}catch(e){alert(e.message)}}init();
</script></body></html>"""


def make_handler(labeler: Labeler):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value: dict, status: int = 200) -> None:
            body = json.dumps(value).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def read_body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:
            try:
                route = urlparse(self.path)
                query = parse_qs(route.query)
                if route.path == "/":
                    body = PAGE.encode("utf-8"); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
                if route.path == "/api/sessions": return self.send_json({"sessions": list(labeler.sessions)})
                key = query.get("key", [""])[0]
                if route.path == "/api/state": return self.send_json(labeler.state(key))
                if route.path == "/preview":
                    image = labeler.preview(key, int(query.get("ic", ["0"])[0]))
                    self.send_response(200); self.send_header("Content-Type", "image/png"); self.send_header("Content-Length", str(len(image))); self.end_headers(); self.wfile.write(image); return
                self.send_json({"error": "not found"}, 404)
            except Exception as exc: self.send_json({"error": str(exc)}, 400)

        def do_POST(self) -> None:
            try:
                data = self.read_body()
                if self.path == "/api/label": return self.send_json(labeler.set_label(data["key"], int(data["ic"]), data["label"]))
                if self.path == "/api/export": return self.send_json({"path": str(labeler.export(data["key"]))})
                self.send_json({"error": "not found"}, 404)
            except Exception as exc: self.send_json({"error": str(exc)}, 400)

        def log_message(self, *_args) -> None: pass
    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ica-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    app = Labeler(args.ica_root, args.label_root)
    print(f"Indexed {len(app.sessions)} sessions. Open http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(app)).serve_forever()
