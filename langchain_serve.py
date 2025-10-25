# this is lang serve ecosystem example code

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI
from langserve import add_routes
import uvicorn


load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)

parser = StrOutputParser()

system_template = "Translate the following into {language}:"

prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

chain = prompt_template | llm | parser


app = FastAPI(
  title="simpleTranslator",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)




# localhost:8000/chain/playground