"""Configuration management for LLMStat.

Loads settings from environment variables and .env files.
"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load .env from project root or current directory
_project_root = Path(__file__).resolve().parent.parent
_dotenv_path = _project_root / ".env"
if _dotenv_path.exists():
    load_dotenv(_dotenv_path)
else:
    load_dotenv()


Backend = Literal["localai", "openai"]


class Config:
    """Singleton configuration loaded from environment."""

    # --- LLM Backend ---
    backend: Backend = os.getenv("LLMSTAT_BACKEND", "localai")  # type: ignore[assignment]

    # --- LocalAI ---
    localai_base_url: str = os.getenv("LOCALAI_BASE_URL", "http://localhost:8080/v1")
    localai_model: str = os.getenv("LOCALAI_MODEL", "llama-3.2-3b-instruct")
    localai_api_key: str = os.getenv("LOCALAI_API_KEY", "not-needed")

    # --- OpenAI ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # --- Study defaults ---
    default_conditions: int = int(os.getenv("DEFAULT_CONDITIONS", "2"))
    default_alpha: float = float(os.getenv("DEFAULT_ALPHA", "0.05"))
    default_equivalence_margin: float = float(os.getenv("DEFAULT_EQUIVALENCE_MARGIN", "0.02"))
    default_min_effect: float = float(os.getenv("DEFAULT_MIN_EFFECT", "0.05"))

    # --- Computed ---
    @property
    def active_base_url(self) -> str:
        if self.backend == "localai":
            return self.localai_base_url
        return self.openai_base_url

    @property
    def active_model(self) -> str:
        if self.backend == "localai":
            return self.localai_model
        return self.openai_model

    @property
    def active_api_key(self) -> str:
        if self.backend == "localai":
            return self.localai_api_key
        return self.openai_api_key

    def validate(self) -> list[str]:
        """Check configuration validity. Returns list of issues (empty = valid)."""
        issues: list[str] = []
        if self.backend not in ("localai", "openai"):
            issues.append(f"Unknown backend: {self.backend}")
        if self.backend == "openai" and not self.openai_api_key:
            issues.append("OPENAI_API_KEY is required when backend=openai")
        return issues


# Global config instance
config = Config()
