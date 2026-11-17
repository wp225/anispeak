from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests


class VllmManager:
    def __init__(self, ask_ai_cfg: Any):
        self._cfg = ask_ai_cfg
        self._proc: subprocess.Popen[str] | None = None
        self._started_here = False
        self._health_url = f"http://{ask_ai_cfg.host}:{ask_ai_cfg.port}/v1/models"
        self._validate_cfg()
        atexit.register(self.stop)

    def _validate_cfg(self) -> None:
        assert Path(self._cfg.model_path).exists(), f"Ask AI model_path does not exist: {self._cfg.model_path}"
        assert str(self._cfg.model_name).strip(), "ask_ai.model_name must be non-empty"
        assert 0 < float(self._cfg.gpu_memory_utilization) <= 1, "ask_ai.gpu_memory_utilization must be in (0, 1]"
        assert int(self._cfg.max_model_len) > 0, "ask_ai.max_model_len must be > 0"
        assert int(self._cfg.max_num_seqs) > 0, "ask_ai.max_num_seqs must be > 0"
        if getattr(self._cfg, "gpu_device", None) is not None:
            assert str(self._cfg.gpu_device).strip(), "ask_ai.gpu_device must be non-empty when provided"

    def is_healthy(self) -> bool:
        try:
            r = requests.get(self._health_url, timeout=5)
            ok = bool(r.ok)
            print(f"[INFO] vLLM health check url={self._health_url} healthy={ok}")
            return ok
        except Exception:
            print(f"[INFO] vLLM health check url={self._health_url} healthy=False")
            return False

    def start(self) -> None:
        if self.is_healthy():
            print("[INFO] Reusing existing healthy vLLM server.")
            return
        if not bool(self._cfg.manage_vllm):
            raise RuntimeError("Ask AI is enabled but vLLM is unreachable and manage_vllm=false.")

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(self._cfg.model_path),
            "--served-model-name",
            str(self._cfg.model_name),
            "--host",
            str(self._cfg.host),
            "--port",
            str(self._cfg.port),
            "--dtype",
            str(self._cfg.dtype),
            "--max-model-len",
            str(self._cfg.max_model_len),
            "--gpu-memory-utilization",
            str(self._cfg.gpu_memory_utilization),
            "--max-num-seqs",
            str(self._cfg.max_num_seqs),
        ]
        env = os.environ.copy()
        gpu_device = getattr(self._cfg, "gpu_device", None)
        if gpu_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
        print("[INFO] Starting vLLM command:", " ".join(cmd))
        print(
            f"[INFO] vLLM launch settings gpu_device="
            f"{gpu_device if gpu_device is not None else '<inherit>'} "
            f"gpu_memory_utilization={self._cfg.gpu_memory_utilization}"
        )
        self._proc = subprocess.Popen(cmd, text=True, env=env)
        self._started_here = True
        print(f"[INFO] Started vLLM PID={self._proc.pid}")

    def wait_until_ready(self, timeout_sec: int) -> None:
        deadline = time.time() + int(timeout_sec)
        while time.time() < deadline:
            if self.is_healthy():
                return
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(f"vLLM exited early with code {self._proc.returncode}")
            time.sleep(2)
        raise TimeoutError(f"Timed out waiting for vLLM to become healthy at {self._health_url}")

    def stop(self) -> None:
        if not self._started_here or self._proc is None:
            return
        if self._proc.poll() is not None:
            return
        print(f"[INFO] Stopping vLLM PID={self._proc.pid}")
        try:
            self._proc.terminate()
            self._proc.wait(timeout=20)
        except Exception:
            try:
                os.kill(self._proc.pid, signal.SIGKILL)
            except Exception:
                pass
