# speciesnet_server.py
import os, json, tempfile, pathlib, subprocess, time
from flask import Flask, request, jsonify
import cv2, numpy as np

app = Flask(__name__)
COUNTRY = os.environ.get("SN_COUNTRY", "USA")
REGION  = os.environ.get("SN_REGION",  "CA")

def run_speciesnet_on_image(bgr_img: np.ndarray):
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="sn_req_"))
    img_path = tmpdir / "frame.jpg"
    cv2.imwrite(str(img_path), bgr_img)
    out_json = tmpdir / "speciesnet_results.json"
    cmd = [
        "python","-m","speciesnet.scripts.run_model",
        "--folders", str(tmpdir),
        "--predictions_json", str(out_json),
        "--country", COUNTRY, "--admin1_region", REGION
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    latency = time.time() - t0
    with open(out_json, "r") as f:
        data = json.load(f)
    preds = data.get("predictions", [])
    best = None
    if preds:
        p = preds[0]
        best = {"label": p.get("classification_label") or p.get("best_label"), "raw": p}
    return best, latency

@app.route("/infer", methods=["POST"])
def infer():
    if "image" not in request.files:
        return jsonify({"error":"missing 'image'"}), 400
    camera = request.form.get("camera","unknown")
    nparr = np.frombuffer(request.files["image"].read(), np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"error":"bad image"}), 400
    try:
        best, latency = run_speciesnet_on_image(bgr)
        return jsonify({"camera": camera, "latency_sec": round(latency,3), "result": best})
    except subprocess.CalledProcessError as e:
        return jsonify({"error":"speciesnet failed","stderr": e.stderr.decode("utf-8","ignore")}), 500

if __name__ == "__main__":
    print("✅ SpeciesNet server on http://0.0.0.0:8000")
    app.run(host="0.0.0.0", port=8000, threaded=True)
