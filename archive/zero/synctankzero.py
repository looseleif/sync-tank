#!/usr/bin/env python3
"""
DEEPLINK: YOLO first, then SpeciesNet (only if fish detected with high confidence).

POST /infer
  form-data:
    - image: file
    - camera: string (optional)
query params:
  - pretty=1 (optional)

Env knobs:
  AUTH_TOKEN         -> if set, require header X-Auth-Token
  SN_COUNTRY         -> default 'USA'
  SN_REGION          -> default 'CA'
  YOLO_WEIGHTS       -> default './weights.pt'
  CONF_FISH          -> default 0.35   (fish threshold)
  DOWNSCALE_MAX_SIDE -> default 1280   (pre-SpeciesNet)
  JPEG_QUALITY       -> default 90

Run:
  python synctankzero.py --host 0.0.0.0 --port 8000
"""

import os, io, json, time, tempfile, pathlib, subprocess, argparse
import numpy as np
import cv2
from flask import Flask, request, jsonify, Response
from ultralytics import YOLO

# ---------------- Config ----------------
AUTH_TOKEN   = os.environ.get("AUTH_TOKEN", "")
SN_COUNTRY   = os.environ.get("SN_COUNTRY", "USA")
SN_REGION    = os.environ.get("SN_REGION",  "CA")
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "./weights.pt")
CONF_FISH    = float(os.environ.get("CONF_FISH", "0.35"))
MAX_SIDE     = int(os.environ.get("DOWNSCALE_MAX_SIDE", "1280"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "90"))

# --------------- YOLO load --------------
_yolo = YOLO(YOLO_WEIGHTS)

# --------------- Flask app --------------
def make_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

    # ---------- helpers ----------
    def _unauth():
        return jsonify({"error": "unauthorized"}), 401

    def _check_auth(req):
        if not AUTH_TOKEN:
            return None
        if req.headers.get("X-Auth-Token") != AUTH_TOKEN:
            return _unauth()
        return None

    def _decode_image_bytes(req):
        if "image" in req.files:
            return req.files["image"].read()
        raw = req.get_data()
        return raw if raw else None

    def _find_fish(result, fish_name="fish", threshold=CONF_FISH):
        """
        Returns True if any detection labeled 'fish' has conf >= threshold.
        `result` is an Ultralytics result object for one image.
        """
        if not result or not result.boxes:
            return False
        names = result.names or {}
        for b in result.boxes:
            conf = float(b.conf.cpu().numpy().item())
            cls  = int(b.cls.cpu().numpy().item())
            label = names.get(cls, str(cls)).lower()
            if label == fish_name and conf >= threshold:
                return True
        return False

    def _speciesnet_top_label_on_frame(frame_bgr):
        """
        Downscale + JPEG to temp → call SpeciesNet CLI once → parse top label
        (ignoring 'blank'/'no cv result').

        Returns dict: {"label": str|None, "score": float|None, "latency_ms": int}
        """
        # Downscale if needed
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, MAX_SIDE / max(h, w))
        if scale < 1.0:
            frame_bgr = cv2.resize(frame_bgr, (int(w*scale), int(h*scale)))

        tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="sn_req_"))
        img_path = tmpdir / "frame.jpg"
        out_json = tmpdir / "out.json"
        cv2.imwrite(str(img_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

        cmd = [
            "python", "-m", "speciesnet.scripts.run_model",
            "--folders", str(tmpdir),
            "--predictions_json", str(out_json),
            "--country", SN_COUNTRY, "--admin1_region", SN_REGION
        ]
        t0 = time.time()
        subprocess.run(cmd, check=True)
        latency = time.time() - t0

        with open(out_json) as f:
            data = json.load(f)
        preds = data.get("predictions", [])
        if not preds:
            return {"label": None, "score": None, "latency_ms": int(latency * 1000)}

        p = preds[0]
        classes = (p.get("classifications") or {}).get("classes", [])
        scores  = (p.get("classifications") or {}).get("scores", [])

        # Pair and filter
        pairs = []
        for cls, sc in zip(classes, scores):
            try:
                sc = float(sc)
            except Exception:
                continue
            tail = (cls.split(";")[-1] if isinstance(cls, str) else "").strip().lower()
            if tail in ("blank", "no cv result"):
                continue
            pairs.append((cls, sc))
        if not pairs:
            return {"label": None, "score": None, "latency_ms": int(latency * 1000)}

        pairs.sort(key=lambda x: x[1], reverse=True)
        top_cls, top_sc = pairs[0]
        label = (top_cls.split(";")[-1] or "unknown").strip()
        return {"label": label, "score": round(float(top_sc), 4), "latency_ms": int(latency * 1000)}

    # ---------- routes ----------
    @app.get("/health")
    def health():
        return {
            "ok": True,
            "yolo_weights": YOLO_WEIGHTS,
            "conf_fish": CONF_FISH,
            "country": SN_COUNTRY,
            "region": SN_REGION,
        }

    @app.get("/")
    def root():
        return "DEEPLINK server ready. POST image to /infer"

    @app.post("/infer")
    def infer():
        # auth
        unauthorized = _check_auth(request)
        if unauthorized:
            return unauthorized

        # read data
        body = _decode_image_bytes(request)
        if not body:
            return jsonify({"error": "no image provided"}), 400
        camera = request.form.get("camera", "unknown")
        want_pretty = request.args.get("pretty", "0") == "1"

        # decode image
        arr = np.frombuffer(body, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "failed to decode image"}), 400

        # YOLO first
        t0 = time.time()
        yres = _yolo(frame, verbose=False)
        y_latency = time.time() - t0
        fish_hit = _find_fish(yres[0])

        out = {
            "camera": camera,
            "yolo_latency_ms": int(y_latency * 1000),
            "fish_detected": bool(fish_hit),
            "species": None,          # filled if fish_detected
            "species_score": None,    # filled if fish_detected
            "species_latency_ms": None
        }

        # Only call SpeciesNet if confident fish
        if fish_hit:
            try:
                s = _speciesnet_top_label_on_frame(frame)
                out["species"] = s.get("label")
                out["species_score"] = s.get("score")
                out["species_latency_ms"] = s.get("latency_ms")
            except subprocess.CalledProcessError as e:
                return jsonify({"error": "speciesnet failed", "detail": str(e)}), 500

        if want_pretty:
            return Response(json.dumps(out, indent=2), mimetype="application/json")
        return jsonify(out)

    return app

# --------------- main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    app = make_app()
    print(f"✅ DEEPLINK on http://{args.host}:{args.port}")
    print("   Flow: YOLO (fish >= {thr}) → SpeciesNet only-if-fish".format(thr=CONF_FISH))
    print("   Env: AUTH_TOKEN, SN_COUNTRY, SN_REGION, YOLO_WEIGHTS, CONF_FISH, DOWNSCALE_MAX_SIDE, JPEG_QUALITY")
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == "__main__":
    main()
