from __future__ import annotations

import io
from pathlib import Path

from fastapi import UploadFile

from app.config import Settings
from app.services.voice_manager import VoiceManager


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        model_dir=tmp_path / "models",
        config_path=tmp_path / "models" / "config.yaml",
        voices_dir=tmp_path / "voices",
        output_dir=tmp_path / "output",
        metadata_path=tmp_path / "voices" / "voices.json",
        default_voice_source=Path("tests/sample_prompt.wav"),
        autoload_model=False,
    )


def test_seed_default_voice(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = VoiceManager(settings)
    voices = list(manager.list_voices())
    assert len(voices) == 1
    assert voices[0].id == settings.default_voice_id


def test_add_voice_creates_unique_ids(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = VoiceManager(settings)

    sample_bytes = Path("tests/sample_prompt.wav").read_bytes()
    upload1 = UploadFile(filename="voice.wav", file=io.BytesIO(sample_bytes))
    voice1 = manager.add_voice(upload1, "Custom Voice")

    upload2 = UploadFile(filename="voice.wav", file=io.BytesIO(sample_bytes))
    voice2 = manager.add_voice(upload2, "Custom Voice")

    assert voice1.id != voice2.id
    stored_paths = {voice1.path, voice2.path}
    for rel_path in stored_paths:
        assert (settings.voices_dir / rel_path).exists()
