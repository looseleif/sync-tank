#!/usr/bin/env python3
"""Small operator agent for the three-Pi Sync Tank wired fleet.

Runs from the display/index node. It treats this machine as local and talks to
tank nodes over SSH on the PRIVATE_IP/24 wired network.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib import error, request


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "fleet_nodes.json"
DEFAULT_SNAPSHOT_DIR = ROOT / "storage" / "fleet_snapshots"


class FleetError(RuntimeError):
    pass


def load_config(path: Path) -> list[dict]:
    try:
      data = json.loads(path.read_text())
    except FileNotFoundError as exc:
      raise FleetError(f"Config not found: {path}") from exc
    nodes = data.get("nodes", [])
    if not isinstance(nodes, list):
        raise FleetError("Config must contain a nodes list")
    return nodes


def node_label(node: dict) -> str:
    return f"{node.get('id')} ({node.get('user')}@{node.get('host')}, {node.get('role')})"


def select_nodes(nodes: list[dict], wanted: list[str] | None) -> list[dict]:
    if not wanted:
        return nodes
    aliases = {
        "one": "tank-one",
        "1": "tank-one",
        "tank1": "tank-one",
        "tank-1": "tank-one",
        "two": "tank-two",
        "2": "tank-two",
        "tank2": "tank-two",
        "tank-2": "tank-two",
        "display": "zero",
        "sync": "zero",
        "local": "zero",
    }
    wanted_ids = {aliases.get(item, item) for item in wanted}
    selected = [node for node in nodes if node.get("id") in wanted_ids]
    missing = wanted_ids - {node.get("id") for node in selected}
    if missing:
        raise FleetError(f"Unknown node id(s): {', '.join(sorted(missing))}")
    return selected


def ssh_prefix(node: dict, batch: bool = False) -> list[str]:
    key_path = Path(os.environ.get("SYNC_TANK_FLEET_KEY", str(Path.home() / ".ssh" / "sync_tank_fleet_ed25519")))
    prefix = [
        "ssh",
        "-o",
        "ConnectTimeout=5",
        "-o",
        "ServerAliveInterval=5",
        "-o",
        "ServerAliveCountMax=2",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    if key_path.exists():
        prefix.extend(["-i", str(key_path)])
    if batch:
        prefix.extend(["-o", "BatchMode=yes"])
    prefix.append(f"{node['user']}@{node['host']}")
    return prefix


def run_local(command: str, timeout: float | None = 20) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-lc", command],
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def run_remote(node: dict, command: str, batch: bool = False, timeout: float | None = 25) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*ssh_prefix(node, batch=batch), command],
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def run_on_node(node: dict, command: str, batch: bool = False, timeout: float | None = 25) -> subprocess.CompletedProcess:
    if node.get("transport") == "local":
        return run_local(command, timeout=timeout)
    return run_remote(node, command, batch=batch, timeout=timeout)


def print_result(node: dict, result: subprocess.CompletedProcess, show_command: str | None = None) -> None:
    header = f"===== {node_label(node)}"
    if show_command:
        header += f" :: {show_command}"
    print(header)
    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print("--- stderr ---")
        print(result.stderr.rstrip())
    print(f"--- exit {result.returncode} ---")


def quote_many(values: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(value)) for value in values)


def repo_discovery_command(node: dict) -> str:
    candidates = node.get("repo_candidates") or []
    quoted = quote_many(candidates)
    return f"""
set -u
echo "host=$(hostname)"
echo "user=$(id -un)"
echo "date=$(date -Is)"
echo "wired_ip={shlex.quote(str(node.get('wired_ip', 'unknown')))}"
for path in {quoted}; do
  if [ -d "$path" ]; then
    echo "repo_candidate=$path"
    if [ -d "$path/.git" ]; then
      git -C "$path" rev-parse --show-toplevel 2>/dev/null | sed 's/^/git_root=/'
      git -C "$path" rev-parse --short HEAD 2>/dev/null | sed 's/^/git_head=/'
      git -C "$path" status --short 2>/dev/null | sed 's/^/git_status=/'
    fi
  fi
done
if command -v find >/dev/null 2>&1; then
  find "$HOME" -maxdepth 3 -type d \\( -name sync-tank -o -name SyncTank -o -name sync_tank \\) 2>/dev/null | sed 's/^/found_repo_dir=/'
fi
""".strip()


def status_command(node: dict) -> str:
    service_lines = []
    for service in node.get("services") or []:
        service_lines.append(
            f"systemctl is-active {shlex.quote(service)} 2>/dev/null | sed 's/^/service {service}=/' || true"
        )
    services = "\n".join(service_lines)
    return f"""
set -u
echo "host=$(hostname)"
echo "user=$(id -un)"
echo "date=$(date -Is)"
echo "kernel=$(uname -srmo)"
echo "uptime=$(uptime -p 2>/dev/null || true)"
echo "ip_addrs=$(ip -brief address 2>/dev/null | tr '\\n' ';')"
echo "routes=$(ip route 2>/dev/null | tr '\\n' ';')"
{services}
""".strip()


def file_tree_command(repo_path: str, max_depth: int) -> str:
    repo = shlex.quote(repo_path)
    return f"""
set -u
if [ ! -d {repo} ]; then
  echo "missing_repo={repo_path}"
  exit 2
fi
find {repo} -maxdepth {int(max_depth)} -type f \
  -not -path '*/.git/*' \
  -not -path '*/.venv/*' \
  -not -path '*/__pycache__/*' \
  | sort
""".strip()


def first_existing_repo(node: dict, batch: bool) -> str | None:
    candidates = node.get("repo_candidates") or []
    if not candidates:
        return None
    test = " ; ".join(
        f"if [ -d {shlex.quote(path)} ]; then echo {shlex.quote(path)}; exit 0; fi"
        for path in candidates
    )
    result = run_on_node(node, test + " ; exit 1", batch=batch, timeout=12)
    if result.returncode == 0:
        return result.stdout.strip().splitlines()[0]
    return None


def get_json(url: str, timeout: float = 4) -> tuple[int | None, dict | None, str]:
    try:
        with request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            try:
                return response.status, json.loads(body), ""
            except json.JSONDecodeError:
                return response.status, None, body[:500]
    except error.URLError as exc:
        return None, None, str(exc)


def command_status(nodes: list[dict], args: argparse.Namespace) -> int:
    failed = 0
    for node in nodes:
        try:
            result = run_on_node(node, status_command(node), batch=args.batch, timeout=args.timeout)
        except (subprocess.SubprocessError, OSError) as exc:
            failed += 1
            print(f"===== {node_label(node)}")
            print(f"error={exc}")
            continue
        if result.returncode:
            failed += 1
        print_result(node, result, "status")
    return 1 if failed else 0


def command_discover(nodes: list[dict], args: argparse.Namespace) -> int:
    failed = 0
    for node in nodes:
        try:
            result = run_on_node(node, repo_discovery_command(node), batch=args.batch, timeout=args.timeout)
        except (subprocess.SubprocessError, OSError) as exc:
            failed += 1
            print(f"===== {node_label(node)}")
            print(f"error={exc}")
            continue
        if result.returncode:
            failed += 1
        print_result(node, result, "discover")
    return 1 if failed else 0


def command_tree(nodes: list[dict], args: argparse.Namespace) -> int:
    failed = 0
    for node in nodes:
        repo = args.repo or first_existing_repo(node, batch=args.batch)
        if not repo:
            failed += 1
            print(f"===== {node_label(node)}")
            print("No repo candidate found")
            continue
        result = run_on_node(node, file_tree_command(repo, args.max_depth), batch=args.batch, timeout=args.timeout)
        if result.returncode:
            failed += 1
        print_result(node, result, f"tree {repo}")
    return 1 if failed else 0


def command_run(nodes: list[dict], args: argparse.Namespace) -> int:
    command = " ".join(args.command).strip()
    if not command:
        raise FleetError("run requires a command after --")
    failed = 0
    for node in nodes:
        result = run_on_node(node, command, batch=args.batch, timeout=args.timeout)
        if result.returncode:
            failed += 1
        print_result(node, result, command)
    return 1 if failed else 0


def command_cat(nodes: list[dict], args: argparse.Namespace) -> int:
    failed = 0
    for node in nodes:
        command = f"sed -n {shlex.quote(args.lines)} {shlex.quote(args.path)}"
        result = run_on_node(node, command, batch=args.batch, timeout=args.timeout)
        if result.returncode:
            failed += 1
        print_result(node, result, f"cat {args.path}")
    return 1 if failed else 0


def command_repo_cat(nodes: list[dict], args: argparse.Namespace) -> int:
    failed = 0
    for node in nodes:
        repo = args.repo or first_existing_repo(node, batch=args.batch)
        if not repo:
            failed += 1
            print(f"===== {node_label(node)}")
            print("No repo candidate found")
            continue
        path = str(Path(repo) / args.path).replace("\\", "/")
        command = f"sed -n {shlex.quote(args.lines)} {shlex.quote(path)}"
        result = run_on_node(node, command, batch=args.batch, timeout=args.timeout)
        if result.returncode:
            failed += 1
        print_result(node, result, f"repo-cat {path}")
    return 1 if failed else 0


def command_ls(nodes: list[dict], args: argparse.Namespace) -> int:
    failed = 0
    for node in nodes:
        path = args.path
        if args.repo_relative:
            repo = args.repo or first_existing_repo(node, batch=args.batch)
            if not repo:
                failed += 1
                print(f"===== {node_label(node)}")
                print("No repo candidate found")
                continue
            path = str(Path(repo) / path).replace("\\", "/")
        command = f"ls -la {shlex.quote(path)}"
        result = run_on_node(node, command, batch=args.batch, timeout=args.timeout)
        if result.returncode:
            failed += 1
        print_result(node, result, f"ls {path}")
    return 1 if failed else 0


def command_grep(nodes: list[dict], args: argparse.Namespace) -> int:
    failed = 0
    for node in nodes:
        repo = args.repo or first_existing_repo(node, batch=args.batch)
        if not repo:
            failed += 1
            print(f"===== {node_label(node)}")
            print("No repo candidate found")
            continue
        include = ""
        for pattern in args.include or []:
            include += f" --include={shlex.quote(pattern)}"
        command = (
            f"grep -RIn{include} "
            "--exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ "
            f"{shlex.quote(args.pattern)} {shlex.quote(repo)} | head -{int(args.limit)}"
        )
        result = run_on_node(node, command, batch=args.batch, timeout=args.timeout)
        if result.returncode not in (0, 1):
            failed += 1
        print_result(node, result, f"grep {args.pattern} {repo}")
    return 1 if failed else 0


def command_http(nodes: list[dict], args: argparse.Namespace) -> int:
    failed = 0
    for node in nodes:
        urls = []
        for key in ("payload_url", "fallback_payload_url"):
            if node.get(key):
                urls.append(node[key])
        if node.get("control_url"):
            urls.append(f"{node['control_url'].rstrip('/')}/api/arm")
        if not urls:
            continue
        print(f"===== {node_label(node)} :: http")
        for url in urls:
            status, payload, err = get_json(url, timeout=args.timeout)
            if status is None:
                failed += 1
                print(f"{url} -> error {err}")
                continue
            summary = ""
            if isinstance(payload, dict):
                node_id = payload.get("node_id") or (payload.get("node") or {}).get("node_id")
                cameras = payload.get("cameras") or (payload.get("camera_registration") or {}).get("cameras") or []
                driver = payload.get("driver")
                if node_id:
                    summary += f" node_id={node_id}"
                if cameras:
                    summary += f" cameras={len(cameras)}"
                if driver:
                    summary += f" driver={driver}"
            print(f"{url} -> HTTP {status}{summary}")
    return 1 if failed else 0


def command_snapshot(nodes: list[dict], args: argparse.Namespace) -> int:
    DEFAULT_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    failed = 0
    for node in nodes:
        repo = args.repo or first_existing_repo(node, batch=args.batch)
        if not repo:
            failed += 1
            print(f"===== {node_label(node)}")
            print("No repo candidate found")
            continue
        out_dir = DEFAULT_SNAPSHOT_DIR / f"{timestamp}-{node['id']}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if node.get("transport") == "local":
            command = f"tar -C {shlex.quote(repo)} --exclude=.git --exclude=.venv --exclude='__pycache__' -cf - . | tar -C {shlex.quote(str(out_dir))} -xf -"
            result = run_local(command, timeout=args.timeout)
        else:
            remote = f"tar -C {shlex.quote(repo)} --exclude=.git --exclude=.venv --exclude='__pycache__' -cf - ."
            local = f"tar -C {shlex.quote(str(out_dir))} -xf -"
            result = subprocess.run(
                [*ssh_prefix(node, batch=args.batch), remote],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout,
            )
            if result.returncode == 0:
                extract = subprocess.run(["bash", "-lc", local], input=result.stdout, stderr=subprocess.PIPE, timeout=args.timeout)
                if extract.returncode:
                    result = subprocess.CompletedProcess(result.args, extract.returncode, result.stdout, extract.stderr)
        if result.returncode:
            failed += 1
        print(f"===== {node_label(node)} :: snapshot {repo}")
        if result.stderr:
            stderr = result.stderr.decode("utf-8", errors="replace") if isinstance(result.stderr, bytes) else result.stderr
            print(stderr.rstrip())
        print(f"saved={out_dir}")
        print(f"--- exit {result.returncode} ---")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Operate and inspect the Sync Tank three-Pi wired fleet")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--node", action="append", help="Node id or alias. Repeat for multiple nodes.")
    parser.add_argument("--batch", action="store_true", help="Disable SSH password prompts; useful for automated checks.")
    parser.add_argument("--timeout", type=float, default=25)
    sub = parser.add_subparsers(dest="command_name", required=True)

    sub.add_parser("status", help="Show host, IP, route, and service status")
    sub.add_parser("discover", help="Discover likely repo paths and git state")

    tree = sub.add_parser("tree", help="List files in each node repo")
    tree.add_argument("--repo", help="Repo path to inspect instead of configured candidates")
    tree.add_argument("--max-depth", type=int, default=3)

    run = sub.add_parser("run", help="Run a shell command on selected nodes")
    run.add_argument("command", nargs=argparse.REMAINDER)

    cat = sub.add_parser("cat", help="Print a remote/local file")
    cat.add_argument("path")
    cat.add_argument("--lines", default="1,220p", help="sed line range, default 1,220p")

    repo_cat = sub.add_parser("repo-cat", help="Print a file relative to each node repo")
    repo_cat.add_argument("path")
    repo_cat.add_argument("--repo", help="Repo path to use instead of configured candidates")
    repo_cat.add_argument("--lines", default="1,220p", help="sed line range, default 1,220p")

    ls_cmd = sub.add_parser("ls", help="List a directory on selected nodes")
    ls_cmd.add_argument("path", nargs="?", default=".")
    ls_cmd.add_argument("--repo-relative", action="store_true", help="Treat path as relative to the discovered repo")
    ls_cmd.add_argument("--repo", help="Repo path to use instead of configured candidates")

    grep_cmd = sub.add_parser("grep", help="Search text inside each selected node repo")
    grep_cmd.add_argument("pattern")
    grep_cmd.add_argument("--repo", help="Repo path to search instead of configured candidates")
    grep_cmd.add_argument("--include", action="append", help="File glob to include, such as '*.py'. Repeat as needed.")
    grep_cmd.add_argument("--limit", type=int, default=80)

    sub.add_parser("http", help="Check known payload/control HTTP endpoints from this display node")

    snapshot = sub.add_parser("snapshot", help="Copy a lightweight repo snapshot from selected nodes to storage/fleet_snapshots")
    snapshot.add_argument("--repo", help="Repo path to snapshot instead of configured candidates")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        nodes = select_nodes(load_config(args.config), args.node)
        handlers = {
            "status": command_status,
            "discover": command_discover,
            "tree": command_tree,
            "run": command_run,
            "cat": command_cat,
            "repo-cat": command_repo_cat,
            "ls": command_ls,
            "grep": command_grep,
            "http": command_http,
            "snapshot": command_snapshot,
        }
        return handlers[args.command_name](nodes, args)
    except FleetError as exc:
        print(f"fleet-agent: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
