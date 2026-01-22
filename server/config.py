# server/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Исправлено: MODELS_PATH в верхнем регистре для соответствия main.py
    MODELS_PATH: str = "saved_models"
    
    TRAINING_CORES: int = 4
    
    MAX_INFERENCE_MODELS: int = 1

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

settings = Settings()

