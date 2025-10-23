# rag_lcel.py

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# Load env vars
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Load Documents
loader = TextLoader("data.txt")  # A sample text file with your data
docs = loader.load()

# Split text into chunks for embeddings
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)


# Create Vector Embeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# Setup Groq LLM

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)


# Define LCEL Prompt Chain

template = """You are a helpful assistant. Use the retrieved context to answer the question.

Context:
{context}

Question:
{question}

Answer only from the context.
"""

prompt = ChatPromptTemplate.from_template(template)

# LCEL pipeline (2nd gen)
rag_chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)


# Run a Query

query = "Explain what LangChain Memory is??."
response = rag_chain.invoke(query)

print("\n RAG Response:\n")
print(response.content)
