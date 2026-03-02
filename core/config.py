import os
import json
from pathlib import Path

# Load simple env variables manually since dotenv might not be installed yet in requirements.
# If python-dotenv is added later, we can use load_dotenv()

class Config:
    def __init__(self):
        self.BASE_DIR = Path(os.environ.get("BASE_DIR", "."))
        self.STORAGE_DIR = self.BASE_DIR / os.environ.get("STORAGE_DIR", "storage")
        self.CONFIG_DIR = self.BASE_DIR / os.environ.get("CONFIG_DIR", "config")
        self.LOG_DIR = self.STORAGE_DIR / "logs"

        # Ensure directories exist
        self.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        (self.STORAGE_DIR / "work").mkdir(parents=True, exist_ok=True)
        (self.STORAGE_DIR / "outputs").mkdir(parents=True, exist_ok=True)

    def load_json_config(self, filename: str) -> dict:
        """Loads a JSON configuration file from the centralized config directory."""
        filepath = self.CONFIG_DIR / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

# Global settings instance
settings = Config()
