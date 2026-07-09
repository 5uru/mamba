import json
import time
from pathlib import Path

import jax
import psutil


_LOG_PATH = None
_CONFIG_PATH = None


def init_run(config: dict, save_dir: str = "checkpoints"):
    global _LOG_PATH, _CONFIG_PATH

    log_dir = Path(save_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = log_dir / "metrics.jsonl"
    _CONFIG_PATH = log_dir / "config.json"

    devices = jax.devices()
    device_info = []
    for d in devices:
        device_info.append({
            "platform": d.platform,
            "client": str(d.client),
            "id": d.id,
        })

    full_config = {
        **config,
        "jax_devices": device_info,
        "jax_device_count": len(devices),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(_CONFIG_PATH, "w") as f:
        json.dump(full_config, f, indent=2)

    print(f"Metrics will be saved to: {_LOG_PATH}")


def log_metrics(step: int, metrics: dict, step_time: float = None):
    if _LOG_PATH is None:
        return

    to_log = {"step": step, "timestamp": time.time(), **metrics}

    if step_time is not None:
        to_log["step_time_s"] = step_time

    mem = psutil.virtual_memory()
    to_log["ram_used_gb"] = round(mem.used / (1024**3), 2)
    to_log["ram_percent"] = mem.percent

    process = psutil.Process()
    proc_mem = process.memory_info()
    to_log["process_rss_gb"] = round(proc_mem.rss / (1024**3), 4)

    with open(_LOG_PATH, "a") as f:
        f.write(json.dumps(to_log) + "\n")


def finish():
    print(f"Metrics saved to: {_LOG_PATH}")
