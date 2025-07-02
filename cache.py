# Standard library imports
import os
import json
import hashlib
from pathlib import Path
from typing import Any, Optional

# Local imports
from config import Config
from utils import safe_json_dump, clean_for_json


class CacheManager:
    """Manages caching of intermediate files with user isolation."""

    @staticmethod
    def get_file_path(
        user_id: str, video_path: str, suffix: str, prompt: Optional[str] = None
    ) -> Path:
        """Generate user-specific file path for cached files."""
        base_path = Config.get_intermediate_path(user_id)
        video_basename = Path(video_path).stem

        if prompt and "trailer_segments" in suffix:
            prompt_hash = hashlib.md5(prompt.strip().encode()).hexdigest()[:8]
            filename = f"{video_basename}_trailer_segments_{prompt_hash}{suffix.replace('trailer_segments', '')}"
            return base_path / filename

        filename = f"{video_basename}_{suffix}"
        return base_path / filename

    @staticmethod
    def load_cache(cache_path: Path) -> Optional[Any]:
        """Load data from cache if it exists."""
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text())
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def save_cache(cache_path: Path, data: Any) -> bool:
        """Save data to cache."""
        cleaned_data = clean_for_json(data)
        return safe_json_dump(cleaned_data, cache_path)

    @staticmethod
    def remove_cache(cache_path: Path) -> None:
        """Remove a specific cache file."""
        if cache_path.exists():
            cache_path.unlink()
