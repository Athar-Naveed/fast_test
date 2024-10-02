from memory_profiler import profile
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage  # for message history

from dotenv import load_dotenv
import os
import google.generativeai as genai
from fastapi import HTTPException
load_dotenv()
# Import utility functions
from utils import save_conversation_history, load_conversation_history, load_prompt_from_file
from pathlib import Path

app: FastAPI = FastAPI()

base_url = os.getenv("base_url")
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


try:
    if base_url is None:
        raise HTTPException("404", "No base URL specified")
    elif google_api_key is None:
        raise HTTPException("404", "No Google API Key specified")
    elif groq_api_key is None:
        raise HTTPException("404", "No GroQ API Key specified")
except Exception as e:
    raise HTTPException("500", f"An error occurred: {str(e)}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[base_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
def index():
    return {"message": "Hello, FastAPI!"}

def handle_input(user_id: str, prompt: str):
    from agents.input_agent import InputAgent
    # Initialize shared memory
    shared_memory = ConversationBufferMemory()

    # Load user profile and history as you did before
    file_path = os.path.join(Path("user_data"), f"{user_id}_profile.json")
    if not os.path.exists(file_path):
        # Initialize new profile
        user_profile = {
            "user_name": user_id,  # Use user_id as default name
            "programming_languages": [],
            "learning_goals": [],
            "strengths": [],
            "weaknesses": [],
            "recent_activities": [],
            "schedule": {},
            "challenges": [],
            "interests": [],
            "sentiment": [],
            "unconfirmed": {}
        }
        with open(file_path, "w") as f:
            json.dump(user_profile, f, indent=4)

    # Initialize input agent
    input_agent = InputAgent(shared_memory, user_id, "user_data")

    # Run the agent
    input_agent_output = input_agent.run(prompt)

    # Return result
    return {"input_agent_output": input_agent_output, "shared_memory": shared_memory}



async def solution(user_id: str, prompt: str, analyzer_output: str, shared_memory: dict,llama_llm):
    from agents.solution_agent import SolutionAgent
    solution_agent = SolutionAgent(llama_llm, shared_memory, user_id, "user_data")

    # Generate the solution
    solution_output = solution_agent.run(prompt, analyzer_output, shared_memory)
    return {"solution_output": solution_output, "shared_memory": shared_memory}



@profile()
# Parameters
@app.post("/chat/orchestrate")
async def orchestrate_chat(user_id: str, prompt: str):
    # Create the Llama 3.1 LLM using ChatGroq
    llama_llm = ChatGroq(
        model="llama-3.1-70b-versatile",  # Or the appropriate Llama model
        temperature=0.0,
        max_retries=2,
        api_key=groq_api_key
    )
    # Initialize shared memory
    shared_memory = ConversationBufferMemory()

    # Store the internal dialogue for the full pipeline
    full_internal_dialogue = ""

    # Step 1: Input Agent
    input_response = handle_input(user_id, prompt)
    shared_memory = input_response["shared_memory"]
    full_internal_dialogue += f"### Input Agent\n{input_response['input_agent_output']}\n\n"

    

    # Step 4: Solution Agent
    solution_response = await solution(user_id, prompt, input_response, shared_memory,llama_llm)
    solution_output = solution_response["solution_output"]
    full_internal_dialogue += f"### Solution Agent\n{solution_output}\n\n"

    

    # Gemini model setup
    gemini_system_instructions = load_prompt_from_file(Path("prompts/gemini_prompt.txt"))
    gemini_model = genai.GenerativeModel(
        model_name="models/gemini-1.5-flash-002",  # Or the appropriate Gemini model name
        system_instruction=gemini_system_instructions,
    )


    # Final response from Gemini (or your main chatbot logic)
    gemini_prompt = f"""
    User: {prompt}

    ## Agent's Responses (Use this as a GUIDE. As a Second Brain):
    ```
    {full_internal_dialogue}
    ```

    ## Conversation History (Carefully reply while keeping the flow of the conversation):
    {shared_memory.buffer}

    PathAI:
    """
    gemini_response = gemini_model.generate_content(gemini_prompt).text

    # Save conversation history
    save_conversation_history(user_id, "user_data", shared_memory)

    # Return both the internal dialogue and the final response
    return {
        "final_response": gemini_response,
        "full_internal_dialogue": full_internal_dialogue
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", reload=True)
