import json
import uuid
from typing import Any
from pathlib import Path


def convert_to_mp3(video_path, audio_path):
    import ffmpeg

    try:
        (
            ffmpeg.input(video_path)
            .output(str(audio_path), ar=16000, ac=1, audio_bitrate="64k", format="mp3")
            .run(capture_stderr=True)
        )
        print(f"Audio extracted successfully to: {audio_path}")
    except ffmpeg.Error as e:
        print(f"An error occurred: {e.stderr.decode('utf8')}")


def generate_user_id() -> str:
    """Generate a unique user ID."""
    return str(uuid.uuid4())


def safe_json_dump(data: Any, file_path: Path, indent: int = 2) -> bool:
    """Safely dumps data to JSON with robust error handling."""
    try:
        json.dumps(data)  # Validate JSON serializable
        file_path.write_text(json.dumps(data, indent=indent, ensure_ascii=False))
        return True
    except Exception:
        return False


def clean_for_json(obj: Any) -> Any:
    """Recursively clean an object to ensure it's JSON serializable."""
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)
