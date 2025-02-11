"""Configuration management for the application."""
from pathlib import Path
from typing import Optional
import yaml
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    """Application settings loaded from config file and environment variables."""
    client_id: str
    nats_url: str
    nats_user: str
    nats_password: str
    chroma_db_path: str
    llama_cloud_api_key: Optional[str] = None

    @classmethod
    def from_config(cls, config_path: Optional[Path] = None) -> "Settings":
        """Load settings from config file."""
        if config_path is None:
            # Look for config file in standard locations
            locations = [
                Path.cwd() / "config.yaml",
                Path.home() / ".config" / "whisk" / "config.yaml",
                Path("/etc/whisk/config.yaml")
            ]
            for loc in locations:
                if loc.exists():
                    config_path = loc
                    break
            else:
                raise FileNotFoundError("No config file found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        return cls(
            client_id=config["client"]["id"],
            nats_url=config["nats"]["url"],
            nats_user=config["nats"]["user"],
            nats_password=config["nats"]["password"],
            chroma_db_path=config["chroma"]["path"],
            llama_cloud_api_key=os.getenv("LLAMA_CLOUD_API_KEY", config["llm"].get("cloud_api_key"))
        )

    class Config:
        env_file = ".env"

# Initialize settings from config file
try:
    settings = Settings.from_config()
except FileNotFoundError:
    # Fallback to environment variables if no config file found
    settings = Settings() 