"""Start a local llama-server, run the long Thymos conversation, then stop it.

Why this wrapper exists: the Manus local execution process and the user's
already-running model can live in different loopback namespaces.  Starting the
server as a child of this Python process guarantees the test client and the
server share a network namespace.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Keep the repository portable.  Point these at your own llama.cpp build and
# GGUF model, either through the environment or with --exe/--model.
DEFAULT_EXE = Path(os.environ.get("LLAMA_SERVER_EXE", "llama-server"))
DEFAULT_MODEL = Path(os.environ.get("LLAMA_MODEL", "models/model.gguf"))
LOG = ROOT / ".work" / "longrun" / "server.log"
ERR = ROOT / ".work" / "longrun" / "server.err.log"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def stop_existing() -> None:
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)


def wait_health(proc: subprocess.Popen, timeout_s: int = 150) -> bool:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"server exited early: code={proc.returncode}", flush=True)
            return False
        try:
            with OPENER.open("http://127.0.0.1:8080/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
                last_error = f"HTTP {resp.status}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    print(f"health timeout; last={last_error}", flush=True)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", default=str(DEFAULT_EXE))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--turns", type=int, default=60)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--fresh", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true", help="keep the state file and continue")
    parser.add_argument("--script", default=str(ROOT / "experiments" / "longrun_local.py"))
    parser.add_argument("--script-args", default="")
    args = parser.parse_args()

    LOG.parent.mkdir(parents=True, exist_ok=True)
    stop_existing()
    if not Path(args.exe).exists():
        print(f"missing llama-server: {args.exe}", flush=True)
        return 2
    if not Path(args.model).exists():
        print(f"missing model: {args.model}", flush=True)
        return 2

    cmd = [
        args.exe,
        "-m",
        args.model,
        "-a",
        "qwen3.6-35b-a3b",
        "--n-cpu-moe",
        "34",
        "--fit",
        "on",
        "--fit-target",
        "1024",
        "-c",
        "32768",
        "-fa",
        "on",
        "-t",
        "6",
        "-tb",
        "6",
        "--parallel",
        "1",
        "--jinja",
        "-lm",
        "none",
        "-rea",
        "off",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
    ]
    print("starting local model...", flush=True)
    with LOG.open("w", encoding="utf-8", errors="replace") as log, ERR.open(
        "w", encoding="utf-8", errors="replace"
    ) as err:
        proc = subprocess.Popen(cmd, stdout=log, stderr=err, cwd=str(ROOT))
        try:
            if not wait_health(proc):
                print("model failed to become healthy; log tail:", flush=True)
                print("\n".join(LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]), flush=True)
                return 3
            print("model ready; running conversation", flush=True)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT / "src")
            env["PYTHONIOENCODING"] = "utf-8"
            script = Path(args.script)
            if not script.is_absolute():
                script = ROOT / script
            test_cmd = [sys.executable, "-X", "utf8", str(script)]
            if script.name == "longrun_local.py":
                test_cmd += [
                    "--base-url",
                    "http://127.0.0.1:8080/v1",
                    "--model",
                    "qwen3.6-35b-a3b",
                    "--turns",
                    str(args.turns),
                    "--start",
                    str(args.start),
                    "--max-tokens",
                    str(args.max_tokens),
                    "--temperature",
                    str(args.temperature),
                    "--top-k",
                    str(args.top_k),
                    "--max-chars",
                    str(args.max_chars),
                ]
                if not args.resume:
                    test_cmd.append("--fresh")
            else:
                test_cmd += shlex.split(args.script_args)
                test_cmd += [
                    "--base-url",
                    "http://127.0.0.1:8080/v1",
                    "--model",
                    "qwen3.6-35b-a3b",
                ]
                # The long-run script owns --turns; the ablation script uses
                # its own fixed query set and does not accept it.
                if script.name != "gate_ablation.py":
                    test_cmd += ["--turns", str(args.turns)]
            test = subprocess.run(
                test_cmd,
                cwd=str(ROOT),
                env=env,
                check=False,
            )
            return test.returncode
        finally:
            print("stopping test model...", flush=True)
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force",
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


if __name__ == "__main__":
    raise SystemExit(main())
