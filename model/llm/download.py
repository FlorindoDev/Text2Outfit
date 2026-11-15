"""Resolve local LLM checkpoints, downloading missing files from Hugging Face."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from huggingface_hub.utils._validators import validate_repo_id


def resolve_model_path(
    model_name: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Use a local directory or complete Hub cache before allowing downloads."""
    if not isinstance(model_name, (str, Path)) or not str(model_name).strip():
        raise ValueError("model_name must be a non-empty model ID or local path")
    local_path = Path(model_name).expanduser()
    if local_path.exists():
        if not local_path.is_dir():
            raise NotADirectoryError(f"model path is not a directory: {local_path}")
        _require_checkpoint(local_path)
        return local_path.resolve()
    if isinstance(model_name, Path) or local_path.is_absolute() or str(model_name).startswith(
        ("./", "../", ".\\", "..\\", "~"),
    ):
        raise FileNotFoundError(f"model directory does not exist: {local_path}")

    validate_repo_id(model_name)
    arguments = {"repo_id": model_name, "repo_type": "model", "cache_dir": cache_dir}
    try:
        cached_path = Path(snapshot_download(**arguments, local_files_only=True))
    except LocalEntryNotFoundError:
        pass
    else:
        # A cached snapshot directory may contain metadata but no model weights yet.
        if _checkpoint_complete(cached_path):
            return cached_path.resolve()

    downloaded_path = Path(snapshot_download(**arguments))
    _require_checkpoint(downloaded_path)
    return downloaded_path.resolve()


def _require_checkpoint(path: Path) -> None:
    if not _checkpoint_complete(path):
        raise ValueError(f"incomplete model checkpoint (config, tokenizer or weights): {path}")


def _checkpoint_complete(path: Path) -> bool:
    if not all(_has_file(path, name) for name in ("config.json", "tokenizer_config.json")):
        return False
    has_tokenizer = any(_has_file(path, name) for name in (
        "tokenizer.json", "tokenizer.model", "spiece.model", "vocab.txt",
    )) or all(_has_file(path, name) for name in ("vocab.json", "merges.txt"))
    if not has_tokenizer:
        return False
    for name in (
        "model.safetensors", "model.safetensors.index.json",
        "pytorch_model.bin", "pytorch_model.bin.index.json",
    ):
        if not _has_file(path, name):
            continue
        if not name.endswith(".index.json"):
            return True
        try:
            index = json.loads((path / name).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeError):
            return False
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        if not all(isinstance(shard, str) and shard for shard in weight_map.values()):
            return False
        shards = set(weight_map.values())
        return all(
            not Path(shard).is_absolute()
            and ".." not in Path(shard).parts
            and _has_file(path, shard)
            for shard in shards
        )
    return False


def _has_file(path: Path, name: str) -> bool:
    file_path = path / name
    return file_path.is_file() and file_path.stat().st_size > 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download or reuse a local Hugging Face LLM.")
    parser.add_argument("model_name", help="Hugging Face model ID or existing checkpoint directory")
    parser.add_argument("--cache-dir", type=Path, help="override the standard Hugging Face cache")
    arguments = parser.parse_args(argv)
    print(resolve_model_path(arguments.model_name, cache_dir=arguments.cache_dir))


if __name__ == "__main__":
    main()
