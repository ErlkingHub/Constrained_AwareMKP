"""Local artifact, reproducibility, and checkpoint utilities."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import scipy
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_SCHEMA_VERSION = 1


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def timestamp() -> str:
    """Filesystem-safe local timestamp; manifests additionally record UTC ISO time."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False) + "\n"
    # Same-directory replace preserves old complete metadata on interruption.
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    requested = str(requested)
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are auto, cpu, cuda or cuda:<index>")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {device.index} is unavailable")
    return device


def _git(*arguments: str) -> str:
    try:
        result = subprocess.run(["git", *arguments], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, encoding="utf-8",
                                errors="replace", check=True, timeout=15)
        return result.stdout.rstrip("\n")
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def _source_files() -> list[Path]:
    """Copy only project source/documentation, never run data or environment files."""
    allowed = {".py", ".md", ".toml", ".yaml", ".yml", ".txt", ".json"}
    candidates = set(PROJECT_ROOT.glob("*.py")) | set(PROJECT_ROOT.glob("*.md"))
    for folder in ("docs", "tests", "configs"):
        base = PROJECT_ROOT / folder
        if base.is_dir():
            candidates.update(p for p in base.rglob("*") if p.is_file() and p.suffix in allowed)
    for name in ("requirements.txt", "pyproject.toml", ".gitignore"):
        path = PROJECT_ROOT / name
        if path.is_file():
            candidates.add(path)
    return sorted((p for p in candidates if p.is_file()), key=lambda p: p.as_posix())


def runtime_manifest(config: Any) -> dict[str, Any]:
    diff = _git("diff", "HEAD", "--", ".")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "config": asdict(config) if is_dataclass(config) else config,
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "status": _git("status", "--short", "--untracked-files=all"),
            "tracked_diff": diff,
            "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        },
        "source_sha256": {str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(p)
                          for p in _source_files()},
        "runtime": {
            "python_executable": sys.executable,
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "torch_num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }


def save_source_snapshot(output_dir: str | Path) -> dict[str, str]:
    """Save actual dirty/untracked code, not only hashes of unrecoverable files.

    The caller normally passes ``run_dir / 'source'``. The returned mapping is
    suitable for its manifest; paths are relative to the project root.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for source in _source_files():
        relative = source.relative_to(PROJECT_ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative.as_posix()] = sha256_file(target)
    (destination / "tracked_changes.patch").write_text(_git("diff", "HEAD", "--", "."), encoding="utf-8")
    write_json(destination / "source_sha256.json", hashes)
    return hashes


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Load this project's trusted local training checkpoint (contains RNG state).

    PyTorch pickle checkpoints must not be loaded from an untrusted source.
    weights_only=False is explicit because NumPy RNG/optimizer state is retained.
    """
    payload = torch.load(resolve_path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        actual = payload.get("schema_version") if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"Unsupported checkpoint schema {actual!r}; expected {CHECKPOINT_SCHEMA_VERSION}")
    for required in ("actor", "config"):
        if required not in payload:
            raise ValueError(f"Checkpoint is missing required field {required!r}")
    return payload
