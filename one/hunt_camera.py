#!/usr/bin/env python3
"""
hunt_camera.py

Fast camera hunter:
- supply IPS or CIDRS via env (IPS=192.168.1.10,192.168.1.108),
  or CIDRS=192.168.1.0/24 (defaults to that range if omitted)
- configure PORTS, PATHS, CREDS as env vars (comma-separated)
- runs nmap ping-sweep, port scans, onvif (UDP 3702), then parallel ffprobe checks
- outputs JSON to hunt_results.json
- use LOOP=1 and INTERVAL=<seconds> to keep trying until found

Example:
  IPS=192.168.1.10,192.168.1.108 CIDRS=192.168.1.0/24 PORTS=554,8554 \
  PATHS=/Streaming/Channels/101,/h264/ch1/main/av_stream \
  CREDS=admin:admin,admin:123456 THREADS=32 \
  python3 hunt_camera.py
"""
import os, sys, json, time, subprocess, shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict

# -------- configuration via env (defaults sensible for IP cams) --------
ENV_IPS = [x.strip() for x in os.getenv("IPS", "").split(",") if x.strip()]
ENV_CIDRS = [x.strip() for x in os.getenv("CIDRS", "192.168.1.0/24").split(",") if x.strip()]
ENV_PORTS = [int(x) for x in os.getenv("PORTS", "554,8554,80,443,8000,5544").split(",") if x.strip().isdigit()]
ENV_PATHS = [p.strip() for p in os.getenv("PATHS", (
    "/Streaming/Channels/101,/h264/ch1/main/av_stream,/h264Preview_01_main,"
    "/live,/main,/videoMain,/live/ch00_0,/live/ch00_1"
)).split(",") if p.strip()]
ENV_CREDS = [c.strip() for c in os.getenv("CREDS", "admin:admin,admin:123456,admin:").split(",") if c.strip()]
THREADS = int(os.getenv("THREADS", "24"))
FFPROBE_TIMEOUT = float(os.getenv("FFPROBE_TIMEOUT", "1.5"))
LOOP = os.getenv("LOOP", "0") == "1"
INTERVAL = int(os.getenv("INTERVAL", "15"))  # seconds between loop iterations
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "hunt_results.json")
# ---------------------------------------------------------------------

def run_cmd(cmd: List[str], timeout: float = 10.0) -> Tuple[int, str, str]:
    """Run command, return (rc, stdout, stderr)."""
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"

def nmap_ping_hosts(cidrs: List[str]) -> List[str]:
    """Fast ping-sweep (nmap -sn) to list live hosts in the CIDRs."""
    hosts = []
    for c in cidrs:
        cmd = ["nmap", "-sn", "-T4", c, "-oG", "-"]
        rc, out, err = run_cmd(cmd, timeout=20.0)
        if rc != 0:
            print(f"[WARN] nmap ping-sweep failed for {c}: rc={rc} err={err.strip()}", file=sys.stderr)
            continue
        for line in out.splitlines():
            if line.startswith("Host:"):
                # Host: 192.168.1.108 ()  Status: Up
                parts = line.split()
                if len(parts) >= 2:
                    ip = parts[1]
                    hosts.append(ip)
    return sorted(set(hosts))

def nmap_port_scan(ip: str, ports: List[int]) -> Dict[int,str]:
    """Quick port scan on the ip for specified ports. Returns {port: state}."""
    port_str = ",".join(str(p) for p in ports)
    cmd = ["nmap", "-p", port_str, "-T4", "--max-retries", "1", "--host-timeout", "5s", ip, "-oG", "-"]
    rc, out, err = run_cmd(cmd, timeout=20.0)
    res = {}
    if rc != 0:
        return res
    for line in out.splitlines():
        if "Ports:" in line:
            # e.g. Ports: 80/open/tcp//http///
            # parse each port token
            after = line.split("Ports:")[-1]
            for tok in after.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                parts = tok.split("/")
                try:
                    p = int(parts[0])
                    state = parts[1]
                    res[p] = state
                except Exception:
                    continue
    return res

def nmap_onvif_scan(cidrs: List[str]) -> Dict[str, str]:
    """Check UDP 3702 for ws-discovery (ONVIF) - returns ip->state."""
    found = {}
    for c in cidrs:
        cmd = ["nmap", "-sU", "-p", "3702", "-T4", "--max-retries", "0", "--host-timeout", "3s", c, "-oG", "-"]
        rc, out, err = run_cmd(cmd, timeout=12.0)
        if rc != 0:
            continue
        for line in out.splitlines():
            if "Ports:" in line and "3702/open" in line or "3702/open|filtered" in line:
                # Host: 192.168.1.108 (hostname)   Ports: 3702/open|filtered/udp///
                parts = line.split()
                if len(parts) >= 2:
                    ip = parts[1]
                    found[ip] = "open"
    return found

def build_candidate_ips() -> List[str]:
    if ENV_IPS:
        return sorted(set(ENV_IPS))
    # otherwise ping-sweep provided cidrs
    hosts = nmap_ping_hosts(ENV_CIDRS)
    # include any local secondary IPs you might have added (optional)
    return hosts

def ffprobe_check(url: str, transport: str="tcp", timeout: float = FFPROBE_TIMEOUT) -> bool:
    """
    Quick ffprobe check for v:0 stream.
    transport: "tcp" or "udp"
    """
    # mmicroseconds for -stimeout
    stimeout = str(int(timeout * 1_000_000))
    cmd = [
        "ffprobe", "-v", "error",
        "-rtsp_transport", transport,
        "-stimeout", stimeout,
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_type",
        "-of", "default=nokey=1:noprint_wrappers=1",
        url
    ]
    rc, out, err = run_cmd(cmd, timeout=timeout + 0.5)
    out_combined = (out or "") + (err or "")
    return rc == 0 and "video" in out_combined.lower()

def probe_rtsp_for_ip(ip: str, ports: List[int], paths: List[str], creds: List[str], threads:int=8) -> List[Tuple[str,str,str]]:
    """
    Try credentials×paths×ports for this ip in parallel (ffprobe tcp then udp fallback).
    Returns list of tuples (ip, url, transport) for working urls.
    """
    tests = []
    for port in ports:
        for cred in creds:
            if ":" in cred:
                user, pw = cred.split(":",1)
            else:
                user, pw = cred, ""
            for path in paths:
                p = path if path.startswith("/") else ("/" + path)
                if user != "":
                    url = f"rtsp://{user}:{pw}@{ip}:{port}{p}"
                else:
                    url = f"rtsp://{ip}:{port}{p}"
                tests.append((ip, port, url))
            # also try query-like variants
            if user:
                q1 = f"rtsp://{ip}:{port}/user={user}&password={pw}&channel=1&stream=0.sdp?real_stream"
                q2 = f"rtsp://{ip}:{port}/user={user}_password={pw}_channel=1_stream=0.sdp?real_stream"
                tests.append((ip, port, q1)); tests.append((ip, port, q2))

    results = []
    # parallel ffprobe checks
    with ThreadPoolExecutor(max_workers=threads) as ex:
        fut_map = {}
        for (ip, port, url) in tests:
            fut = ex.submit(ffprobe_check, url, "tcp", FFPROBE_TIMEOUT)
            fut_map[fut] = (ip, port, url, "tcp")
        for fut in as_completed(fut_map):
            ip, port, url, transport = fut_map[fut]
            try:
                ok = fut.result()
            except Exception:
                ok = False
            if ok:
                results.append((ip, url, transport))
            else:
                # schedule a quick udp check (do this inline to avoid doubling all tests up front)
                fut2 = ex.submit(ffprobe_check, url, "udp", FFPROBE_TIMEOUT)
                try:
                    ok2 = fut2.result()
                except Exception:
                    ok2 = False
                if ok2:
                    results.append((ip, url, "udp"))
    # de-duplicate on url
    seen = set(); out = []
    for r in results:
        if r[1] not in seen:
            seen.add(r[1]); out.append(r)
    return out

def probe_http_ui(ip: str, ports: List[int]) -> Dict[int, str]:
    """Try basic HTTP HEAD on common UI ports to detect web UI quickly."""
    res = {}
    for p in [80, 443, 8000, 8080, 8443]:
        if p not in ports:
            continue
        try:
            if p == 443 or p == 8443:
                cmd = ["curl", "-k", "--max-time", "2", "-I", f"https://{ip}:{p}/"]
            else:
                cmd = ["curl", "--max-time", "2", "-I", f"http://{ip}:{p}/"]
            rc, out, err = run_cmd(cmd, timeout=3.0)
            if rc == 0 and out:
                res[p] = "http"
        except Exception:
            continue
    return res

def hunt_once() -> Dict:
    """Run one full hunt iteration and return structured results."""
    summary = {"candidates": [], "rtsp_found": [], "nmap_ports": {}, "onvif": {}, "http_ui": {}}
    ips = build_candidate_ips()
    print(f"[HUNT] candidate IPs: {ips}", flush=True)
    summary["candidates"] = ips

    # port scan + onvif
    for ip in ips:
        ports_state = nmap_port_scan(ip, ENV_PORTS)
        summary["nmap_ports"][ip] = ports_state

    onvif_hits = nmap_onvif_scan(ENV_CIDRS)
    summary["onvif"] = onvif_hits

    # quick HTTP UI checks for ports from nmap results
    for ip in ips:
        ports = list(summary["nmap_ports"].get(ip, {}).keys())
        http = probe_http_ui(ip, ports)
        summary["http_ui"][ip] = http

    # RTSP probing in parallel per IP
    rtsp_results = []
    for ip in ips:
        print(f"[HUNT] probing RTSP on {ip} ...", flush=True)
        res = probe_rtsp_for_ip(ip, ENV_PORTS, ENV_PATHS, ENV_CREDS, threads=max(4, THREADS//2))
        for (ip2, url, transport) in res:
            print(f"[FOUND] {ip2} -> {url} ({transport})", flush=True)
            rtsp_results.append({"ip": ip2, "url": url, "transport": transport})
    summary["rtsp_found"] = rtsp_results

    # write results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=2)
    return summary

def pretty_print_summary(s: Dict):
    print("=== HUNT SUMMARY ===")
    print("Candidates:", s.get("candidates", []))
    print("ONVIF responders:", list(s.get("onvif", {}).keys()))
    print("RTSP found:")
    for r in s.get("rtsp_found", []):
        print(" ", r["ip"], "->", r["url"], f"({r['transport']})")
    print("Nmap ports (sample):")
    for ip, ports in s.get("nmap_ports", {}).items():
        print(" ", ip, ports)
    print("====================")

def main():
    first = True
    while True:
        s = hunt_once()
        pretty_print_summary(s)
        if s.get("rtsp_found"):
            print("[HUNT] Found RTSP streams; stopping because we found at least one.", flush=True)
            break
        if not LOOP:
            print("[HUNT] No RTSP streams found this pass. Exiting (LOOP not enabled).", flush=True)
            break
        print(f"[HUNT] sleeping {INTERVAL}s then re-trying...", flush=True)
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()

