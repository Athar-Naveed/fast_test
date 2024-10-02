from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage  # for message history
# Import agent classes
from agents.input_agent import InputAgent
from agents.analyzer_agent import AnalyzerAgent
from agents.planner_agent import PlannerAgent
from agents.solution_agent import SolutionAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.memory_agent import MemoryAgent
from dotenv import load_dotenv
import os
import google.generativeai as genai
from fastapi import HTTPException
load_dotenv()
# Import utility functions
from utils import save_conversation_history, load_conversation_history, load_prompt_from_file
from pathlib import Path

app:FastAPI = FastAPI()

base_url = os.getenv("base_url")
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
    
    
try:
    if base_url == None:
        raise HTTPException("404","No base URL specified")
    elif google_api_key == None:
        raise HTTPException("404","No Google API Key specified")
    elif groq_api_key == None:
        raise HTTPException("404","No GroQ API Key specified")
except Exception as e:
    raise HTTPException("500", f"An error occurred: {str(e)}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Create the Llama 3.1 LLM using ChatGroq
llama_llm = ChatGroq(
    model="llama-3.1-70b-versatile",  # Or the appropriate Llama model
    temperature=0.0,
    max_retries=2,
    api_key=groq_api_key
)

# Gemini model setup
gemini_system_instructions = load_prompt_from_file(Path("prompts/gemini_prompt.txt"))
gemini_model = genai.GenerativeModel(
    model_name="models/gemini-1.5-flash-002",  # Or the appropriate Gemini model name
    system_instruction=gemini_system_instructions, 
)


@app.get("/")
def index():
    return {"message": "Hello, FastAPI!"}

# Parameters
@app.post("/chat")
async def chat(user_id:str,prompt:str):
    try:
         # Initialize shared memory
        shared_memory = ConversationBufferMemory()

        # --- User Profile and History Loading ---
        file_path = os.path.join(Path("user_data"), f"{user_id}_profile.json")

        # 1. Load (or create) user profile
        if not os.path.exists(file_path): 
            # Create a new profile if it doesn't exist (using a template this time)
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

        #Initialize agents
        input_agent = InputAgent(shared_memory, user_id, "user_data")
        analyzer_agent = AnalyzerAgent(llama_llm, shared_memory, user_id, "user_data")
        planner_agent = PlannerAgent(llama_llm, shared_memory, user_id, "user_data")
        solution_agent = SolutionAgent(llama_llm, shared_memory, user_id, "user_data")
        evaluator_agent = EvaluatorAgent(llama_llm, shared_memory, user_id, "user_data")
        memory_agent = MemoryAgent(llama_llm, shared_memory, user_id, "user_data", file_path)

        # 2. Load conversation history (moved - should be after creating/loading profile and before processing messages)
        load_conversation_history(user_id, "user_data", shared_memory)


        # Get the latest message  
        # prompt = request.messages[-1].content

        # Run the agent pipeline
        input_agent_output = input_agent.run(prompt)
        full_internal_dialogue = f"### Input Agent\n{input_agent_output}\n\n"

        while True:
            analyzer_output = analyzer_agent.run(prompt, shared_memory)
            full_internal_dialogue += f"### Analyzer Agent\n{analyzer_output}\n\n"

            planner_output = planner_agent.run(analyzer_output, shared_memory)
            full_internal_dialogue += f"### Planner Agent\n{planner_output}\n\n"

            solution_output = solution_agent.run(prompt, analyzer_output, planner_output, shared_memory)
            full_internal_dialogue += f"### Solution Agent\n{solution_output}\n\n"

            evaluation_status, evaluator_output = evaluator_agent.run(solution_output, shared_memory)
            full_internal_dialogue += f"### Evaluator Agent\n{evaluator_output}\n\n"

            if evaluation_status == "ACCEPTED":
                break
            elif evaluation_status == "REVISION_REQUESTED":
                full_internal_dialogue += "Solution needs revision. Restarting from Solution Agent...\n\n"

        # Run the Memory Agent
        memory_agent.run(full_internal_dialogue)  # No need to store output


        # --- Gemini's Turn ---
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

        return {"response":gemini_response, "agent_responses":full_internal_dialogue}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", reload=True)