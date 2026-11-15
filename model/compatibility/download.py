"""Reuse OutfitTransformer source/weights locally before downloading missing assets."""

from __future__ import annotations

import argparse
import io
import logging
import tempfile
import urllib.request
import zipfile
from dataclasses import replace
from pathlib import Path, PurePosixPath

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

from ..config import DEFAULT_MODEL_CONFIG, ModelConfig


def resolve_cp_assets(config: ModelConfig = DEFAULT_MODEL_CONFIG) -> tuple[Path, Path]:
    """Return local source and one best.pt; explicit checkpoints must already exist."""
    config.validate()
    checkpoint = config.cp_checkpoint_path.expanduser()
    if config.cp_checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"CP checkpoint not found: {checkpoint}")
    repo = config.cp_repo.expanduser().resolve()
    if not (repo / "model/__init__.py").is_file():
        if repo.exists():
            raise ValueError(f"CP repository exists but has no model/__init__.py: {repo}")
        _download_source(repo, config.cp_source_url)
    if checkpoint.is_file():
        return repo, checkpoint.resolve()

    try:
        checkpoint = Path(hf_hub_download(
            repo_id=config.cp_weights_repo, filename=config.cp_weights_file,
            cache_dir=config.cp_cache.expanduser(), local_files_only=True,
        ))
    except LocalEntryNotFoundError:
        logging.getLogger(__name__).info("Downloading CP weights: %s/%s", config.cp_weights_repo, config.cp_weights_file)
        checkpoint = Path(hf_hub_download(
            repo_id=config.cp_weights_repo, filename=config.cp_weights_file,
            cache_dir=config.cp_cache.expanduser(),
        ))
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise ValueError(f"downloaded CP checkpoint is missing or empty: {checkpoint}")
    return repo, checkpoint.resolve()


def _download_source(repo: Path, url: str) -> None:
    """Keep only model Python files and publish the directory after extraction succeeds."""
    logging.getLogger(__name__).info("Downloading OutfitTransformer source: %s", url)
    with urllib.request.urlopen(url, timeout=30) as response:
        archive_bytes = response.read()
    repo.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".outfit-transformer-", dir=repo.parent) as directory:
        staging = Path(directory) / "source"
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                    raise ValueError("invalid OutfitTransformer archive path")
                if len(path.parts) < 3 or path.parts[1] != "model" or path.suffix != ".py":
                    continue
                target = staging.joinpath(*path.parts[1:])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
        if not (staging / "model/__init__.py").is_file():
            raise ValueError("downloaded OutfitTransformer archive has no model package")
        staging.replace(repo)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download OutfitTransformer model code and one best.pt checkpoint.")
    parser.add_argument("--cp-repo", type=Path, default=DEFAULT_MODEL_CONFIG.cp_repo)
    parser.add_argument("--cp-checkpoint", type=Path, default=DEFAULT_MODEL_CONFIG.cp_checkpoint)
    parser.add_argument("--cp-cache", type=Path, default=DEFAULT_MODEL_CONFIG.cp_cache)
    parser.add_argument("--cp-weights-repo", default=DEFAULT_MODEL_CONFIG.cp_weights_repo)
    parser.add_argument("--cp-weights-file", default=DEFAULT_MODEL_CONFIG.cp_weights_file)
    parser.add_argument("--cp-source-url", default=DEFAULT_MODEL_CONFIG.cp_source_url)
    config = replace(DEFAULT_MODEL_CONFIG, **vars(parser.parse_args(argv)))
    repo, checkpoint = resolve_cp_assets(config)
    print(f"CP source: {repo}\nCP checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
