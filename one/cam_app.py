import os
import signal
import subprocess
from pathlib import Path
from flask import Flask, render_template_string, send_from_directory

# ---- Configure your streams here ----
STREAMS = [
    {
        "name": "cam185_main_tcp",
        "rtsp": "rtsp://admin:123456@192.168.1.185:554/h264Preview_01_main",
        "transport": "tcp",  # tcp recommended for this one
    },
    {
        "name": "cam185_query_tcp",
        "rtsp": "rtsp://192.168.1.185:554/user=admin_password=_channel=1_stream=0.sdp?real_stream",
        "transport": "tcp",
    },
    {
        "name": "cam108_main_udp",
        "rtsp": "rtsp://admin:admin@192.168.1.108:554/h264/ch1/main/av_stream",
        "transport": "udp",  # this cam worked over UDP in probes
    },
]

# HLS output root (served as static files)
OUTDIR = Path("static/streams")
OUTDIR.mkdir(parents=True, exist_ok=True)

ffmpeg_procs = []

def start_ffmpeg_workers():
    """
    Start one ffmpeg process per stream: RTSP -> HLS segments.
    We keep settings light for a Pi 5 and 'copy' video if already H.264.
    """
    global ffmpeg_procs
    for s in STREAMS:
        name = s["name"]
        target_dir = OUTDIR / name
        target_dir.mkdir(parents=True, exist_ok=True)

        # Clean old segments/playlists on restart
        for p in target_dir.glob("*"):
            try:
                p.unlink()
            except Exception:
                pass

        rtsp_transport = s.get("transport", "tcp")
        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-rtsp_transport", rtsp_transport,
            "-fflags", "nobuffer",
            "-rw_timeout", "5000000",  # 5s read timeout
            "-i", s["rtsp"],

            # Try to stream-copy (assumes H.264) to minimize CPU on Pi
            "-c:v", "copy",
            "-an",

            # HLS output
            "-f", "hls",
            "-hls_time", "2",
            "-hls_list_size", "6",
            "-hls_flags", "delete_segments+append_list+independent_segments+omit_endlist",
            "-max_reload", "1",
            "-method", "PUT",

            # Write to local filesystem
            str(target_dir / "index.m3u8")
        ]

        # On some cameras, strict copy can fail. If you see issues, replace "-c:v copy" with a lightweight transcode:
        # "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-g", "24", "-keyint_min", "24", "-pix_fmt", "yuv420p",

        print("Starting FFmpeg for:", name)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        ffmpeg_procs.append(proc)

def stop_ffmpeg_workers():
    global ffmpeg_procs
    for p in ffmpeg_procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass
    ffmpeg_procs = []

app = Flask(__name__, static_folder="static")

PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Pi 5 – RTSP Viewer</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <!-- hls.js from CDN -->
  <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
  <style>
    body { font-family: system-ui, sans-serif; background:#0b0b0b; color:#eee; margin:0; padding:1rem; }
    h1 { font-size: 1.25rem; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap:1rem; }
    .card { background:#111; border:1px solid #222; border-radius:12px; padding:0.75rem; }
    video { width:100%; height:auto; background:#000; border-radius:8px; }
    .name { font-weight:600; margin:0.5rem 0 0.25rem; }
    .src { font-size:0.8rem; color:#aaa; word-break:break-all; }
  </style>
</head>
<body>
  <h1>Pi 5 – RTSP Streams (HLS)</h1>
  <div class="grid">
    {% for s in streams %}
    <div class="card">
      <div class="name">{{ s.name }}</div>
      <div class="src">{{ s.hls }}</div>
      <video id="{{ s.name }}" controls muted playsinline></video>
    </div>
    {% endfor %}
  </div>

<script>
function attachHls(videoId, src) {
  const video = document.getElementById(videoId);
  if (!video) return;
  if (Hls.isSupported()) {
    const hls = new Hls({ liveBackBufferLength: 0 });
    hls.loadSource(src);
    hls.attachMedia(video);
    hls.on(Hls.Events.MANIFEST_PARSED, function() {
      video.play().catch(()=>{});
    });
  } else if (video.canPlayType('application/vnd.apple.mpegURL')) {
    video.src = src;
    video.play().catch(()=>{});
  } else {
    video.outerHTML = '<div style="color:#faa">HLS not supported in this browser.</div>';
  }
}

const streams = {{ streams|tojson }};
streams.forEach(s => attachHls(s.name, s.hls));
</script>
</body>
</html>
"""

@app.route("/")
def index():
    items = []
    for s in STREAMS:
        items.append({
            "name": s["name"],
            "hls": f"/streams/{s['name']}/index.m3u8"
        })
    return render_template_string(PAGE, streams=items)

# Static serving of HLS content
@app.route("/streams/<path:filename>")
def serve_stream(filename):
    return send_from_directory(OUTDIR, filename)

if __name__ == "__main__":
    try:
        start_ffmpeg_workers()
        # Host on all interfaces for LAN viewing; change port if needed
        app.run(host="0.0.0.0", port=8000, threaded=True)
    finally:
        stop_ffmpeg_workers()
