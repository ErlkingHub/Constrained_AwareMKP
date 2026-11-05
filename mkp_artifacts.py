"""Small reproducibility helpers shared by training and evaluation."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

from mkp_config import PROJECT_ROOT


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def instance_sha(inst):
    """Hash values, shapes and dtypes; independent of the tensor device."""
    digest = hashlib.sha256()
    for name in ("p", "r", "c"):
        array = getattr(inst, name).detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def source_provenance():
    def git(*args):
        result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                                text=True, encoding="utf-8", errors="replace")
        return result.stdout.strip() if result.returncode == 0 else None

    untracked = git("ls-files", "--others", "--exclude-standard")
    files, contents = {}, {}
    for name in (untracked or "").splitlines():
        # Generated runs can be intentionally unignored by the user. They already
        # have separate manifests; embedding them recursively bloats source snapshots.
        if name.replace("\\", "/").split("/")[0] in ("runs", "results", "cache", "checkpoints"):
            continue
        path = os.path.join(PROJECT_ROOT, name)
        if os.path.isfile(path):
            files[name] = sha256_file(path)
            # Git's diff omits untracked source. Preserve it alongside the hash so
            # a run made before committing can still be reconstructed.
            if os.path.splitext(name)[1].lower() in (".py", ".md", ".json", ".toml", ".txt", ".yml", ".yaml"):
                with open(path, "r", encoding="utf-8", errors="replace") as stream:
                    contents[name] = stream.read()
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--short"),
            "diff": git("diff", "HEAD", "--binary"), "untracked_sha256": files,
            "untracked_source": contents}


def write_json(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
