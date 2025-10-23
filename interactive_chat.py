# main.py

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Initialize the LLM (Groq)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=api_key
)

# Initial system prompt
system_prompt = SystemMessage(content="You are the best LangChain explainer in the world. Be concise and clear.")

print("Interactive Chat Started! Type 'exit' or 'quit' to stop.\n")

# Conversation loop
chat_history = [system_prompt]

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Chat ended.")
        break

    chat_history.append(HumanMessage(content=user_input))

    # Invoke the model
    response = llm.invoke(chat_history)
    print(f"AI: {response.content}\n")

    # Append assistant response to maintain context
    chat_history.append(response)
