import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    # Get the project root directory
    PROJECT_ROOT = Path(__file__).parent.parent

    # Constants and Configuration
    UPLOAD_FOLDER = PROJECT_ROOT / "upload_files"
    INTERMEDIATE_FOLDER = PROJECT_ROOT / "intermediate_files"
    DEFAULT_VIDEO_PATH = UPLOAD_FOLDER / "default/s1-e1-01.mp4"
    DEFAULT_PROMPT = "Compile the most peaceful and warm moments."
    DEFAULT_LOGO = "./assets/logo1.jpg"

    # Ensure intermediate folder exists
    INTERMEDIATE_FOLDER.mkdir(parents=True, exist_ok=True)

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file"
        )

    @classmethod
    def setup_openai(cls):
        """Configure OpenAI with API key."""
        import openai

    @classmethod
    def get_intermediate_path(cls, user_id: str) -> Path:
        """Get user-specific intermediate folder path."""
        user_folder = cls.INTERMEDIATE_FOLDER / user_id
        user_folder.mkdir(exist_ok=True)
        return user_folder

    @classmethod
    def cleanup_old_files(cls, max_age_hours: int = 24):
        """
        Clean up old intermediate files.

        Args:
            max_age_hours: Maximum age of files in hours before deletion
        """
        import time

        current_time = time.time()

        try:
            for user_folder in cls.INTERMEDIATE_FOLDER.iterdir():
                if user_folder.is_dir():
                    # Check each file in the user's folder
                    for file_path in user_folder.iterdir():
                        file_age = current_time - file_path.stat().st_mtime
                        if file_age > (max_age_hours * 3600):
                            file_path.unlink(missing_ok=True)

                    # Remove empty user folders
                    if not any(user_folder.iterdir()):
                        user_folder.rmdir()
        except Exception as e:
            print(f"Error during cleanup: {e}")
