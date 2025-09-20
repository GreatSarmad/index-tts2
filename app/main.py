"""FastAPI application exposing Index-TTS2 inference endpoints."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import Settings
from app.models import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    VoiceCloneResponse,
    VoiceListResponse,
)
from app.services.tts_service import TTSService
from app.services.voice_manager import VoiceManager

LOGGER = logging.getLogger(__name__)


def create_app(
    *,
    settings: Optional[Settings] = None,
    voice_manager: Optional[VoiceManager] = None,
    tts_service: Optional[TTSService] = None,
) -> FastAPI:
    settings = settings or Settings()
    settings.ensure_directories()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    voice_manager = voice_manager or VoiceManager(settings)
    tts_service = tts_service or TTSService(settings, voice_manager)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        LOGGER.info("Starting Index-TTS2 FastAPI service")
        if settings.autoload_model:
            LOGGER.info("Autoloading TTS model as requested")
            tts_service.ensure_model_loaded()
        yield

    app = FastAPI(title="Index-TTS2 FastAPI", version="2.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.voice_manager = voice_manager
    app.state.tts_service = tts_service

    app.mount(
        "/output",
        StaticFiles(directory=settings.output_dir, check_dir=False),
        name="output",
    )

    def get_voice_manager(request: Request) -> VoiceManager:
        return request.app.state.voice_manager

    def get_tts_service(request: Request) -> TTSService:
        return request.app.state.tts_service

    async def parse_generate_request(
        request: Request,
        text: Optional[str] = Form(default=None),
        voice_id: Optional[str] = Form(default=None),
        speed: float = Form(default=1.0),
        pitch: float = Form(default=0.0),
    ) -> GenerateRequest:
        if text is not None:
            return GenerateRequest(text=text, voice_id=voice_id, speed=speed, pitch=pitch)
        try:
            data = await request.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - FastAPI handles parsing
            raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc
        return GenerateRequest(**data)

    @app.get("/health", response_model=HealthResponse)
    async def health(service: TTSService = Depends(get_tts_service)) -> HealthResponse:
        payload = service.health()
        return HealthResponse(status="ok", **payload)

    @app.get("/voices", response_model=VoiceListResponse)
    async def list_voices(manager: VoiceManager = Depends(get_voice_manager)) -> VoiceListResponse:
        voices = list(manager.list_voices())
        return VoiceListResponse(voices=voices)

    @app.post("/clone-voice", response_model=VoiceCloneResponse)
    async def clone_voice(
        voice_name: str = Form(..., description="Name for the cloned voice"),
        file: UploadFile = File(..., description="Reference audio sample"),
        manager: VoiceManager = Depends(get_voice_manager),
    ) -> VoiceCloneResponse:
        voice = manager.add_voice(file, voice_name)
        return VoiceCloneResponse(voice_id=voice.id, name=voice.name, path=voice.path)

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(
        request: Request,
        payload: GenerateRequest = Depends(parse_generate_request),
        service: TTSService = Depends(get_tts_service),
    ) -> GenerateResponse:
        response = service.generate(payload)
        audio_url = request.url_for("output", path=response.audio_path)
        response.audio_url = str(audio_url)
        response.audio_path = f"output/{response.audio_path}"
        return response

    return app


app = create_app()
