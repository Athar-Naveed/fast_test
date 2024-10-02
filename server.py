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


async def analyze(user_id: str, prompt: str, shared_memory: dict,llama_llm):
    from agents.analyzer_agent import AnalyzerAgent
    analyzer_agent = AnalyzerAgent(llama_llm, shared_memory, user_id, "user_data")

    # Analyze the prompt and return the result
    analyzer_output = analyzer_agent.run(prompt, shared_memory)
    return {"analyzer_output": analyzer_output, "shared_memory": shared_memory}


async def plan(user_id: str, analyzer_output: str, shared_memory: dict,llama_llm):
    from agents.planner_agent import PlannerAgent
    planner_agent = PlannerAgent(llama_llm, shared_memory, user_id, "user_data")

    # Plan based on analyzer output
    planner_output = planner_agent.run(analyzer_output, shared_memory)
    return {"planner_output": planner_output, "shared_memory": shared_memory}


async def solution(user_id: str, prompt: str, analyzer_output: str, shared_memory: dict,llama_llm):
    from agents.solution_agent import SolutionAgent
    solution_agent = SolutionAgent(llama_llm, shared_memory, user_id, "user_data")

    # Generate the solution
    solution_output = solution_agent.run(prompt, analyzer_output, shared_memory)
    return {"solution_output": solution_output, "shared_memory": shared_memory}


async def evaluate(user_id: str, solution_output: str, shared_memory: dict,llama_llm):
    from agents.evaluator_agent import EvaluatorAgent
    evaluator_agent = EvaluatorAgent(llama_llm, shared_memory, user_id, "user_data")

    # Evaluate the solution
    evaluation_status, evaluator_output = evaluator_agent.run(solution_output, shared_memory)
    return {"evaluation_status": evaluation_status, "evaluator_output": evaluator_output, "shared_memory": shared_memory}


async def memory(user_id: str, full_internal_dialogue: str, shared_memory: dict,llama_llm):
    from agents.memory_agent import MemoryAgent
    memory_agent = MemoryAgent(llama_llm, shared_memory, user_id, "user_data", f"user_data/{user_id}_profile.json")

    # Update the memory with internal dialogue
    memory_agent.run(full_internal_dialogue)
    return {"message": "Memory updated", "shared_memory": shared_memory}

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

    # Step 2: Analyzer Agent
    # analyze_response = await analyze(user_id, prompt, shared_memory,llama_llm)
    # analyzer_output = analyze_response["analyzer_output"]
    # full_internal_dialogue += f"### Analyzer Agent\n{analyzer_output}\n\n"

    # Step 3: Planner Agent
    # plan_response = await plan(user_id, analyzer_output, shared_memory,llama_llm)
    # planner_output = plan_response["planner_output"]
    # full_internal_dialogue += f"### Planner Agent\n{planner_output}\n\n"

    # Step 4: Solution Agent
    solution_response = await solution(user_id, prompt, input_response, shared_memory,llama_llm)
    solution_output = solution_response["solution_output"]
    full_internal_dialogue += f"### Solution Agent\n{solution_output}\n\n"

    # Step 5: Evaluator Agent
    eval_response = await evaluate(user_id, solution_output, shared_memory,llama_llm)
    evaluator_output = eval_response["evaluator_output"]
    full_internal_dialogue += f"### Evaluator Agent\n{evaluator_output}\n\n"

    if eval_response["evaluation_status"] == "REVISION_REQUESTED":
        full_internal_dialogue += "Solution needs revision. Restarting from Solution Agent...\n\n"
        # If revisions are requested, loop back to Solution Agent

    # Step 6: Memory Agent
    await memory(user_id, full_internal_dialogue, shared_memory,llama_llm)

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
