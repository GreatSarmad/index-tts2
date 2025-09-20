"""Voice prompt management utilities."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, Optional

from fastapi import HTTPException, UploadFile

from app.config import Settings
from app.models import Voice


_SLUGIFY_RE = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    slug = _SLUGIFY_RE.sub("-", value.lower()).strip("-")
    return slug or f"voice-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"


class VoiceManager:
    """Persists voice prompts on disk and exposes metadata."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = RLock()
        self._voices: Dict[str, Voice] = {}
        self.bootstrap()

    @property
    def metadata_path(self) -> Path:
        return self.settings.metadata_path

    def bootstrap(self) -> None:
        """Ensure directories and default voices exist."""

        self.settings.ensure_directories()
        with self._lock:
            self._load()
            if not self._voices:
                self._seed_default_voice()

    def _load(self) -> None:
        if not self.metadata_path.exists():
            self._voices = {}
            return

        data = json.loads(self.metadata_path.read_text())
        voices = {}
        for item in data.get("voices", []):
            try:
                voice = Voice(**item)
            except Exception:
                continue
            voice_path = (self.settings.voices_dir / voice.path).resolve()
            if voice_path.exists():
                voices[voice.id] = voice
        self._voices = voices

    def _save(self) -> None:
        payload = {"voices": [voice.model_dump() for voice in self._voices.values()]}
        tmp_path = self.metadata_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, default=str, indent=2))
        tmp_path.replace(self.metadata_path)

    def _seed_default_voice(self) -> None:
        source = self.settings.default_voice_source
        if not source:
            return
        source_path = (Path(source).resolve())
        if not source_path.exists():
            return
        dest_dir = self.settings.voices_dir / self.settings.default_voice_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / source_path.name
        shutil.copy2(source_path, dest_file)
        self._voices[self.settings.default_voice_id] = Voice(
            id=self.settings.default_voice_id,
            name=self.settings.default_voice_name,
            path=str(dest_file.relative_to(self.settings.voices_dir)),
            created_at=datetime.now(timezone.utc),
        )
        self._save()

    def list_voices(self) -> Iterable[Voice]:
        with self._lock:
            return list(self._voices.values())

    def get_voice(self, voice_id: Optional[str]) -> Voice:
        with self._lock:
            if voice_id is None:
                voice_id = self.settings.default_voice_id
            voice = self._voices.get(voice_id)
            if not voice:
                raise HTTPException(status_code=404, detail="Voice not found")
            return voice

    def register_voice(self, *, voice_id: str, name: str, file_path: Path) -> Voice:
        rel_path = file_path.relative_to(self.settings.voices_dir)
        voice = Voice(id=voice_id, name=name, path=str(rel_path), created_at=datetime.now(timezone.utc))
        with self._lock:
            self._voices[voice.id] = voice
            self._save()
        return voice

    def _resolve_target(self, voice_id: str, filename: str) -> Path:
        target_dir = self.settings.voices_dir / voice_id
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / filename

    def add_voice(self, upload: UploadFile, voice_name: str) -> Voice:
        if not voice_name.strip():
            raise HTTPException(status_code=400, detail="voice_name must not be empty")

        slug = _slugify(voice_name)
        candidate = slug
        with self._lock:
            counter = 1
            while candidate in self._voices:
                counter += 1
                candidate = f"{slug}-{counter}"
        voice_id = candidate

        suffix = Path(upload.filename or "voice.wav").suffix or ".wav"
        filename = f"prompt{suffix}"
        target_file = self._resolve_target(voice_id, filename)
        upload.file.seek(0)
        with target_file.open("wb") as file_obj:
            shutil.copyfileobj(upload.file, file_obj)

        voice = self.register_voice(voice_id=voice_id, name=voice_name.strip(), file_path=target_file)
        return voice

    def absolute_voice_path(self, voice: Voice) -> Path:
        path = (self.settings.voices_dir / voice.path).resolve()
        if not path.exists():
            raise HTTPException(status_code=404, detail="Voice prompt file missing")
        return path
