from pydantic import BaseSettings

class Settings(BaseSettings):

    #openai
    OPENAI_API_KEY : str

    # vectordb
    PERSIST_DIRECTORY : str = "app/vector_db/"


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()