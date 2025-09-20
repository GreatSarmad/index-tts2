"""Application configuration management."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="INDEXTTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
    )

    model_dir: Path = Field(default=Path("/app/models"), description="Directory containing Index-TTS2 checkpoints.")
    config_path: Path = Field(default=Path("/app/models/config.yaml"), description="Path to the model configuration file.")
    voices_dir: Path = Field(default=Path("/app/voices"), description="Directory where speaker prompts are stored.")
    output_dir: Path = Field(default=Path("/app/output"), description="Directory where generated audio is stored.")
    metadata_path: Path = Field(default=Path("/app/voices/voices.json"), description="Voice metadata JSON path.")
    default_voice_name: str = Field(default="Default", description="Display name for the seeded default voice.")
    default_voice_id: str = Field(default="default", description="Identifier for the default voice.")
    default_voice_source: Optional[Path] = Field(
        default=Path("examples/voice_01.wav"),
        description="Relative path to the seed voice sample bundled with the repository.",
    )
    cache_dir: Path = Field(default=Path("/app/models/hf_cache"), description="Cache directory for Hugging Face downloads.")
    modelscope_cache_dir: Path = Field(
        default=Path("/app/models/modelscope_cache"),
        description="Cache directory for ModelScope downloads.",
    )
    autoload_model: bool = Field(default=False, description="Whether to load the TTS model during startup.")
    device: Optional[str] = Field(default=None, description="Optional override for torch device string.")
    use_fp16: bool = Field(default=False, description="Enable fp16 inference when CUDA is available.")
    use_cuda_kernel: bool = Field(default=True, description="Enable custom CUDA kernels when running on GPU.")
    use_deepspeed: bool = Field(default=False, description="Enable DeepSpeed optimization if installed.")
    max_voice_duration: int = Field(default=15, description="Maximum length in seconds for stored voice prompts.")
    log_level: str = Field(default="INFO", description="Application logging level.")

    def ensure_directories(self) -> None:
        """Create directories required by the application."""

        for path in (self.model_dir, self.voices_dir, self.output_dir, self.cache_dir, self.modelscope_cache_dir):
            path.mkdir(parents=True, exist_ok=True)
        if self.metadata_path.parent:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
