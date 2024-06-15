from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langserve import add_routes
from app.config import settings
from dotenv import load_dotenv
from app.multiquery_chain import chain
load_dotenv()

app = FastAPI(
    title="rag_dev",
    version="1.0",
    description="rag_with_langchain",
)

add_routes(
    app,
    ChatOpenAI(model="gpt-4o",
               temperature=0.2),
    path="/chat",
)

add_routes(
    app,
    chain,
    path="/rag_chat",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)