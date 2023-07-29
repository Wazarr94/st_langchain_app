from pathlib import Path

from pydantic import BaseSettings


class Settings(BaseSettings):
    HUGGINGFACEHUB_API_TOKEN: str

    class Config:
        env_file = ".env"


settings = Settings()  # pyright: ignore

project_path = Path(__file__).parent.parent
package_path = project_path / "langchain_stuff"
