"""Application configuration and thresholds."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / ".chroma"


def load_environment() -> None:
    """Load local environment files without overwriting existing variables."""
    for env_file in (PROJECT_ROOT / ".env", PROJECT_ROOT / "API.env"):
        if load_dotenv:
            load_dotenv(env_file, override=False)
        else:
            _load_env_file(env_file)


@dataclass(frozen=True)
class AppConfig:
    gemini_api_key: str | None
    chat_model: str = "gemini-1.5-flash"
    embedding_model: str = "text-embedding-004"
    collection_name: str = "support_knowledge"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 4
    min_retrieval_score: float = 0.10
    escalation_score: float = 0.06
    max_context_chars: int = 5000


def get_config() -> AppConfig:
    load_environment()
    return AppConfig(
        gemini_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
    )


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


PERSONA_LABELS = ("Tech", "Frustrated", "Exec")

SENSITIVE_KEYWORDS = {
    "refund",
    "chargeback",
    "legal",
    "lawsuit",
    "breach",
    "security incident",
    "compromised",
    "cancel account",
    "delete account",
    "payment failed",
}
