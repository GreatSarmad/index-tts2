"""Utility script to pre-download IndexTTS2 model assets."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download

LOGGER = logging.getLogger(__name__)


def _download_hf(repo_id: str, local_dir: Path, cache_dir: Path, revision: str | None = None) -> None:
    LOGGER.info("Downloading %s to %s", repo_id, local_dir)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        cache_dir=cache_dir,
        revision=revision,
        resume_download=True,
    )


def _download_modelscope(model_id: str, target_dir: Path) -> None:
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("ModelScope is unavailable: %s", exc)
        return

    LOGGER.info("Downloading %s via ModelScope", model_id)
    ms_snapshot_download(model_id=model_id, local_dir=target_dir, cache_dir=str(target_dir))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download IndexTTS2 checkpoints for Docker builds")
    parser.add_argument("--model-dir", type=Path, default=Path("/app/models"), help="Destination directory")
    parser.add_argument(
        "--repo-id",
        default="IndexTeam/IndexTTS-2",
        help="Hugging Face repository containing the checkpoints",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision to download",
    )
    parser.add_argument(
        "--download-auxiliary",
        action="store_true",
        help="Also pre-download auxiliary models (feature extractors, vocoders, etc.)",
    )
    parser.add_argument(
        "--skip-modelscope",
        action="store_true",
        help="Skip ModelScope downloads even when the library is available.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_dir = args.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = model_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    token = os.getenv("HF_TOKEN")
    if token:
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", token)

    _download_hf(args.repo_id, model_dir, cache_dir, revision=args.revision)

    if args.download_auxiliary:
        auxiliaries = [
            "facebook/w2v-bert-2.0",
            "funasr/campplus",
            "amphion/MaskGCT",
        ]
        for repo in auxiliaries:
            target = cache_dir / repo.replace("/", "_")
            target.mkdir(parents=True, exist_ok=True)
            _download_hf(repo, target, cache_dir)

    if not args.skip_modelscope:
        _download_modelscope("IndexTeam/IndexTTS-2", model_dir)


if __name__ == "__main__":
    main()
