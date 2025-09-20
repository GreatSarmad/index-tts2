"""Pydantic models used by the FastAPI service."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Voice(BaseModel):
    """Metadata describing an available speaker voice."""

    id: str = Field(..., description="Unique identifier for the voice.")
    name: str = Field(..., description="Human readable voice name.")
    path: str = Field(..., description="Relative file path to the stored audio prompt.")
    created_at: datetime = Field(..., description="Timestamp when the voice was registered.")


class VoiceListResponse(BaseModel):
    """Response schema for the `/voices` endpoint."""

    voices: list[Voice]


class VoiceCloneResponse(BaseModel):
    """Response schema for `/clone-voice`."""

    voice_id: str = Field(..., description="Identifier that can be reused with `/generate`.")
    name: str = Field(..., description="Display name for the cloned voice.")
    path: str = Field(..., description="Relative storage path for the cloned voice sample.")


class GenerateRequest(BaseModel):
    """Payload accepted by the `/generate` endpoint."""

    text: str = Field(..., description="Text that should be synthesised.")
    voice_id: Optional[str] = Field(
        default=None,
        description="Optional voice identifier. Uses the default speaker when omitted.",
    )
    speed: float = Field(default=1.0, gt=0.25, le=4.0, description="Playback speed multiplier.")
    pitch: float = Field(
        default=0.0,
        ge=-12.0,
        le=12.0,
        description="Pitch shift measured in semitones. Negative values reduce pitch.",
    )


class GenerateResponse(BaseModel):
    """Response payload returned by `/generate`."""

    voice_id: str
    audio_path: str = Field(..., description="Path to the generated audio relative to the application root.")
    audio_url: str = Field(..., description="Absolute URL to the generated audio file.")


class HealthGPU(BaseModel):
    available: bool
    name: Optional[str]
    count: int
    memory_total_gb: Optional[float]


class HealthModel(BaseModel):
    loaded: bool
    device: Optional[str]
    version: Optional[str]


class HealthResponse(BaseModel):
    status: str
    gpu: HealthGPU
    model: HealthModel
    voices: int
