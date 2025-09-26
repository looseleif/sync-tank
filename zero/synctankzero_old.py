#!/usr/bin/env python3
# SpeciesNet link: single-file server (WSL) + sender (Pi)
# - Server: accepts images, runs SpeciesNet via official CLI, returns clean JSON
# - Send  : posts a single image to the server (use on Pi or anywhere)
#
# Deps on SERVER (WSL venv):
#   pip install flask requests speciesnet
#   pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
#   pip install opencv-python-headless numpy
#
# Deps on PI (sender):
#   pip install requests
#
# Usage:
#   # Server (WSL, GPU)
#   python synctankzero.py server --host 0.0.0.0 --port 8000
#
#   # Sender (Pi or any machine)
#   python synctankzero.py send --file /path/to/frame.jpg --url http://100.x.y.z:8000/infer --camera synctank3
#   # If you secure the server:
#   # export AUTH_TOKEN="supersecret"   (on server)
#   # python synctankzero.py send ... --token supersecret
#
# Env knobs (server):
#   AUTH_TOKEN=<string>          # require X-Auth-Token header
#   SN_COUNTRY=USA               # geofilter country (default USA)
#   SN_REGION=CA                 # geofilter admin1 region (default CA)
#   MAX_UPLOAD_MB=8              # max upload size
#   DOWNSCALE_MAX_SIDE=1280      # max image side before inference
#   JPEG_QUALITY=90              # saved temp JPEG quality (1..100)

import os, io, time, json, tempfile, pathlib, subprocess, argparse, sys
from typing import Optional, Tuple, Dict, Any

# --------- Client (Pi) ---------
def send_image(url: str, image_path: str, camera: str, token: Optional[str], timeout: float) -> Tuple[int, dict | str]:
    import requests
    headers = {}
    if token:
        headers["X-Auth-Token"] = token
    with open(image_path, "rb") as f:
        files = {"image": (os.path.basename(image_path), f, "application/octet-stream")}
        data = {"camera": camera}
        r = requests.post(url, headers=headers, data=data, files=files, timeout=timeout)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text

# --------- Server helpers ---------
def _check_auth(req, auth_token: str):
    from flask import jsonify
    if not auth_token:
        return None
    if req.headers.get("X-Auth-Token") != auth_token:
        return jsonify({"error": "unauthorized"}), 401
    return None

def _decode_upload(req) -> Optional[bytes]:
    # Prefer multipart file named "image"; otherwise accept raw body
    if "image" in req.files:
        return req.files["image"].read()
    raw = req.get_data()
    return raw if raw else None

def _gpu_stats() -> Dict[str, Any]:
    # Best-effort: requires nvidia-smi on the WSL host
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            timeout=1.0
        )
        s = out.decode().strip().split(",")
        util = int(s[0].strip()); mem_used = int(s[1].strip()); mem_total = int(s[2].strip())
        return {"gpu_util": util, "gpu_mem_mb": mem_used, "gpu_mem_total_mb": mem_total}
    except Exception:
        return {}

def _speciesnet_cli_on_bytes(img_bytes: bytes, country: str, region: str) -> Tuple[Optional[Dict[str, Any]], float, Dict[str, Any]]:
    # Decode + (optionally) downscale + write temp JPEG, then call SpeciesNet CLI once.
    import cv2
    import numpy as np

    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="sn_req_"))
    img_path = tmpdir / "frame.jpg"
    out_json = tmpdir / "speciesnet_results.json"

    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("failed to decode image")

    max_side = int(os.environ.get("DOWNSCALE_MAX_SIDE", "1280"))
    h, w = frame.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    q = int(os.environ.get("JPEG_QUALITY", "90"))
    cv2.imwrite(str(img_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])

    cmd = [
        "python", "-m", "speciesnet.scripts.run_model",
        "--folders", str(tmpdir),
        "--predictions_json", str(out_json),
        "--country", country, "--admin1_region", region,
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    latency = time.time() - t0

    with open(out_json) as f:
        data = json.load(f)
    preds = data.get("predictions", [])
    if not preds:
        return None, latency, {}
    return preds[0], latency, _gpu_stats()

def _simplify_classes(p: Dict[str, Any], topk=3, threshold=0.01):
    """Turn SpeciesNet class arrays into clean list; skip 'blank' / 'no cv result'."""
    classes = (p.get("classifications") or {}).get("classes", [])
    scores  = (p.get("classifications") or {}).get("scores", [])
    pairs = []
    for cls, sc in zip(classes, scores):
        try:
            scf = float(sc)
        except Exception:
            continue
        tail = (cls.split(";")[-1] if isinstance(cls, str) else "").strip().lower()
        if tail in ("blank", "no cv result"):
            continue
        if scf < float(threshold):
            continue
        pairs.append((cls, scf))
    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = pairs[:int(topk)]

    out = []
    for cls, sc in pairs:
        parts = cls.split(";")
        label = parts[-1] or (parts[-2] if len(parts) >= 2 else "unknown")
        out.append({
            "label": label,
            "score": round(sc, 4),
            "taxonomy": [p for p in parts[1:-1] if p],  # skip UUID and final label
            "raw": cls
        })
    return out

def _simplify_detections(p: Dict[str, Any], maxk=5, min_conf=0.05):
    dets = p.get("detections") or []
    items = []
    for d in dets:
        try:
            conf = float(d.get("conf", 0.0))
        except Exception:
            conf = 0.0
        if conf < min_conf:
            continue
        items.append({
            "label": d.get("label"),
            "conf": round(conf, 4),
            "bbox_xywh_norm": [round(float(v), 4) for v in (d.get("bbox") or [])]
        })
    items.sort(key=lambda x: x["conf"], reverse=True)
    return items[:maxk]

# --------- Flask factory ---------
def make_app():
    from flask import Flask, request, jsonify, Response
    app = Flask(__name__)

    max_mb = int(os.environ.get("MAX_UPLOAD_MB", "8"))
    app.config["MAX_CONTENT_LENGTH"] = max_mb * 1024 * 1024

    AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")
    COUNTRY = os.environ.get("SN_COUNTRY", "USA")
    REGION  = os.environ.get("SN_REGION",  "CA")

    @app.get("/")
    def root():
        return "SpeciesNet server OK. POST an image to /infer"

    @app.get("/health")
    def health():
        info = {"ok": True, "country": COUNTRY, "region": REGION}
        info.update(_gpu_stats())
        return info

    @app.post("/infer")
    def infer():
        # auth
        unauthorized = _check_auth(request, AUTH_TOKEN)
        if unauthorized:
            return unauthorized

        # upload
        body = _decode_upload(request)
        if not body:
            return jsonify({"error": "no image provided"}), 400

        # format controls
        topk = request.args.get("topk", default=3, type=int)
        threshold = request.args.get("threshold", default=0.01, type=float)
        verbose = request.args.get("verbose", default=0, type=int) == 1
        pretty = request.args.get("pretty", default=0, type=int) == 1

        try:
            p, latency, stats = _speciesnet_cli_on_bytes(body, COUNTRY, REGION)  # p is the raw prediction dict
            if p is None:
                clean = {
                    "camera": request.form.get("camera", "unknown"),
                    "latency_ms": int(round(latency * 1000)),
                    "model": "unknown",
                    "label": None,
                    "score": None,
                    "topk": [],
                    "detections": [],
                    "stats": stats or {}
                }
                return jsonify(clean)

            classes = _simplify_classes(p, topk=topk, threshold=threshold)
            detections = _simplify_detections(p, maxk=5, min_conf=0.05)
            best = classes[0] if classes else None

            clean = {
                "camera": request.form.get("camera", "unknown"),
                "latency_ms": int(round(latency * 1000)),
                "model": p.get("model_version") or "unknown",
                "label": (best or {}).get("label"),
                "score": (best or {}).get("score"),
                "topk": classes,
                "detections": detections,
                "stats": stats or {}
            }
            if verbose:
                clean["raw"] = p

            if pretty:
                return Response(json.dumps(clean, indent=2), mimetype="application/json")
            return jsonify(clean)

        except subprocess.CalledProcessError as e:
            return jsonify({"error": "speciesnet failed", "detail": str(e)}), 500
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return app

# --------- Entrypoint ---------
def main():
    ap = argparse.ArgumentParser(description="SpeciesNet link (server on WSL, sender on Pi)")
    sub = ap.add_subparsers(dest="mode", required=True)

    sp_server = sub.add_parser("server", help="Run HTTP server (WSL GPU)")
    sp_server.add_argument("--host", default="0.0.0.0")
    sp_server.add_argument("--port", type=int, default=8000)

    sp_send = sub.add_parser("send", help="Send one image to server (Pi or any host)")
    sp_send.add_argument("--file", required=True, help="Image path (jpg/png)")
    sp_send.add_argument("--url", required=True, help="Server /infer URL (e.g., http://100.x.y.z:8000/infer)")
    sp_send.add_argument("--camera", default="pi")
    sp_send.add_argument("--token", default=None, help="Auth token if server set AUTH_TOKEN")
    sp_send.add_argument("--timeout", type=float, default=120.0)

    args = ap.parse_args()

    if args.mode == "server":
        app = make_app()
        print(f"✅ SpeciesNet server on http://{args.host}:{args.port}")
        print("   Env knobs: AUTH_TOKEN, SN_COUNTRY, SN_REGION, MAX_UPLOAD_MB, DOWNSCALE_MAX_SIDE, JPEG_QUALITY")
        # Note: Flask dev server is fine for LAN testing. For production, use gunicorn/uvicorn.
        app.run(host=args.host, port=args.port, threaded=True)

    elif args.mode == "send":
        if not os.path.isfile(args.file):
            print(f"ERROR: file not found: {args.file}", file=sys.stderr)
            sys.exit(2)
        code, resp = send_image(args.url, args.file, args.camera, args.token, args.timeout)
        if code != 200:
            print(f"HTTP {code}\n{resp}", file=sys.stderr)
            sys.exit(3)
        # friendly print
        try:
            camera = resp.get("camera")
            label = (resp.get("label"))
            latency_ms = resp.get("latency_ms")
            print(f"OK [{camera}] {os.path.basename(args.file)} -> label={label}  latency={latency_ms}ms")
            print(json.dumps(resp, indent=2))
        except Exception:
            print(resp)

if __name__ == "__main__":
    main()
