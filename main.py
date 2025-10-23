# main.py

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # You can use any Groq-supported model (like mixtral-8x7b)
    api_key=api_key
)

# Create a message list
messages = [
("system","Assume you are the best langchain explainer in the world."),
("human","Explain LangChain in simple words"),
]

# Get model response
response = llm.invoke(messages)

print("Model Response:\n")
print(response.content)
