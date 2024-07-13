import os

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_chroma import Chroma
from config import settings
from dotenv import load_dotenv
load_dotenv()

# embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def check_folder_empty(folder_path):

    # check if folder exists
    if not os.path.exists(folder_path) or not os.listdir(folder_path):
        
        print(f"the folder {folder_path} does not exist or does not contain any vectorstore.")
        print("loading and splitting data.")

        # load
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
        data = loader.load()
        
        # split
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)

        print("creating vectorstore.")
        
        # init vectordb
        vector_db = Chroma.from_documents(
            documents=all_splits, 
            embedding=embeddings,
            persist_directory=settings.PERSIST_DIRECTORY)
        retriever = vector_db.as_retriever()
    
    else:
        
        # load persisted vectordb
        vector_db = Chroma(persist_directory=settings.PERSIST_DIRECTORY, embedding_function=embeddings)
        retriever = vector_db.as_retriever()

    print("vectorstore ready.")
    return retriever
        

# Set up index with multi query retriever
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
retriever = MultiQueryRetriever.from_llm(retriever=check_folder_empty(folder_path=settings.PERSIST_DIRECTORY), llm=llm)

# RAG prompt
template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
model = ChatOpenAI()
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
