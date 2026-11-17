import os
import subprocess
import sys

from anispeakarchive.core.config import load_settings
from anispeakarchive.core.env import load_dotenv
from scripts.vllm_manager import VllmManager


def main():
    # Ensure env-driven secrets (SMTP, etc.) are available in dev.
    load_dotenv(".env")

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        print("ERROR: CONDA_PREFIX not set. Activate your conda env first.", file=sys.stderr)
        sys.exit(1)

    os.environ["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_PRELOAD"] = f"{conda_prefix}/lib/libstdc++.so.6:{conda_prefix}/lib/libgcc_s.so.1:" + os.environ.get("LD_PRELOAD", "")

    settings = load_settings("configs/default.yaml")
    if settings.ask_ai.enabled and settings.ask_ai.provider == "vllm_openai":
        mgr = VllmManager(settings.ask_ai)
        mgr.start()
        mgr.wait_until_ready(settings.ask_ai.startup_timeout_sec)

    # run as module so imports work
    cmd = [sys.executable, "-m", "app"]
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
