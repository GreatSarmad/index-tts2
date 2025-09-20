"""Wrapper around the IndexTTS2 inference pipeline."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import HTTPException

from app.config import Settings
from app.models import GenerateRequest, GenerateResponse
from app.utils.audio import apply_speed_and_pitch
from app.services.voice_manager import VoiceManager

LOGGER = logging.getLogger(__name__)


class TTSService:
    """Manage model lifecycle and expose synthesis helpers."""

    def __init__(self, settings: Settings, voice_manager: VoiceManager):
        self.settings = settings
        self.voice_manager = voice_manager
        self._lock = Lock()
        self._model = None
        self._model_version: Optional[str] = None
        self._device: Optional[str] = None

    def ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return

            try:
                from indextts.infer_v2 import IndexTTS2
            except ImportError as exc:
                raise HTTPException(status_code=500, detail=f"Failed to import IndexTTS2: {exc}") from exc

            os.environ.setdefault("HF_HOME", str(self.settings.cache_dir))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(self.settings.cache_dir))
            os.environ.setdefault("MODELSCOPE_CACHE", str(self.settings.modelscope_cache_dir))
            os.environ.setdefault("HF_HUB_CACHE", str(self.settings.cache_dir))

            self.settings.ensure_directories()
            LOGGER.info("Loading IndexTTS2 from %s", self.settings.model_dir)

            model = IndexTTS2(
                cfg_path=str(self.settings.config_path),
                model_dir=str(self.settings.model_dir),
                use_fp16=self.settings.use_fp16,
                device=self.settings.device,
                use_cuda_kernel=self.settings.use_cuda_kernel,
                use_deepspeed=self.settings.use_deepspeed,
            )
            self._model = model
            self._device = getattr(model, "device", None)
            self._model_version = getattr(model, "model_version", None)
            LOGGER.info("IndexTTS2 loaded on %s", self._device)

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def model_version(self) -> Optional[str]:
        return self._model_version

    def device(self) -> Optional[str]:
        return self._device

    def generate(self, payload: GenerateRequest) -> GenerateResponse:
        if not payload.text or not payload.text.strip():
            raise HTTPException(status_code=400, detail="text must not be empty")

        voice = self.voice_manager.get_voice(payload.voice_id)
        voice_path = self.voice_manager.absolute_voice_path(voice)

        self.ensure_model_loaded()
        if self._model is None:
            raise HTTPException(status_code=500, detail="Model failed to load")

        output_dir = self.settings.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        unique_id = uuid.uuid4().hex
        raw_output = output_dir / f"{unique_id}_raw.wav"
        final_output = output_dir / f"{unique_id}.wav"

        try:
            generated_path = self._model.infer(
                spk_audio_prompt=str(voice_path),
                text=payload.text,
                output_path=str(raw_output),
            )
        except Exception as exc:  # pragma: no cover - surfaced to client
            LOGGER.exception("TTS inference failed")
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

        if not generated_path or not Path(generated_path).exists():
            raise HTTPException(status_code=500, detail="Model did not return a valid audio path")

        speed = float(payload.speed)
        pitch = float(payload.pitch)
        if abs(speed - 1.0) > 1e-3 or abs(pitch) > 1e-3:
            try:
                apply_speed_and_pitch(Path(generated_path), final_output, speed=speed, pitch=pitch)
            finally:
                Path(generated_path).unlink(missing_ok=True)
        else:
            Path(generated_path).replace(final_output)

        return GenerateResponse(
            voice_id=voice.id,
            audio_path=str(final_output.relative_to(self.settings.output_dir)),
            audio_url="",
        )

    def health(self) -> dict:
        import torch

        gpu_available = torch.cuda.is_available()
        gpu_name = None
        memory_total = None
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return {
            "gpu": {
                "available": gpu_available,
                "name": gpu_name,
                "count": torch.cuda.device_count(),
                "memory_total_gb": round(memory_total, 2) if memory_total else None,
            },
            "model": {
                "loaded": self.is_model_loaded(),
                "device": self.device(),
                "version": self.model_version(),
            },
            "voices": len(list(self.voice_manager.list_voices())),
        }
