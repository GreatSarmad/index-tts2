from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import GenerateRequest, GenerateResponse
from app.services.voice_manager import VoiceManager


class StubTTSService:
    def __init__(self, settings: Settings, voice_manager: VoiceManager) -> None:
        self.settings = settings
        self.voice_manager = voice_manager
        self.calls: list[GenerateRequest] = []

    def ensure_model_loaded(self) -> None:  # pragma: no cover - not used in tests
        pass

    def is_model_loaded(self) -> bool:
        return True

    def model_version(self) -> str:
        return "stub"

    def device(self) -> str:
        return "cpu"

    def generate(self, payload: GenerateRequest) -> GenerateResponse:
        self.calls.append(payload)
        voice = self.voice_manager.get_voice(payload.voice_id)
        output_dir = self.settings.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "test.wav"
        out_file.write_bytes(b"RIFF")
        return GenerateResponse(voice_id=voice.id, audio_path=out_file.name, audio_url="")

    def health(self) -> Dict[str, Any]:
        return {
            "gpu": {"available": False, "name": None, "count": 0, "memory_total_gb": None},
            "model": {"loaded": True, "device": "cpu", "version": "stub"},
            "voices": len(list(self.voice_manager.list_voices())),
        }


def make_client(tmp_path: Path) -> tuple[TestClient, StubTTSService]:
    settings = Settings(
        model_dir=tmp_path / "models",
        config_path=tmp_path / "models" / "config.yaml",
        voices_dir=tmp_path / "voices",
        output_dir=tmp_path / "output",
        metadata_path=tmp_path / "voices" / "voices.json",
        default_voice_source=Path("tests/sample_prompt.wav"),
        autoload_model=False,
    )
    voice_manager = VoiceManager(settings)
    service = StubTTSService(settings, voice_manager)
    app = create_app(settings=settings, voice_manager=voice_manager, tts_service=service)
    return TestClient(app), service


def test_list_voices(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    response = client.get("/voices")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["voices"]) == 1


def test_clone_voice(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    sample = Path("tests/sample_prompt.wav").read_bytes()
    files = {"file": ("voice.wav", sample, "audio/wav")}
    data = {"voice_name": "New Voice"}
    response = client.post("/clone-voice", data=data, files=files)
    assert response.status_code == 200
    result = response.json()
    assert result["voice_id"].startswith("new-voice")


def test_generate_with_json_payload(tmp_path: Path) -> None:
    client, service = make_client(tmp_path)
    response = client.post("/generate", json={"text": "hello", "speed": 1.2, "pitch": -1})
    assert response.status_code == 200
    body = response.json()
    assert body["audio_path"].startswith("output/")
    assert body["audio_url"].endswith("/output/test.wav")
    assert service.calls[0].speed == 1.2


def test_generate_with_form_payload(tmp_path: Path) -> None:
    client, service = make_client(tmp_path)
    response = client.post(
        "/generate",
        data={"text": "hello", "speed": 0.9, "pitch": 0.5},
    )
    assert response.status_code == 200
    assert service.calls[0].speed == 0.9


def test_health_endpoint(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["voices"] == 1
