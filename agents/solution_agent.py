import json 
import os
from langchain_groq import ChatGroq 
from utils import load_prompt_from_file
from langchain.memory import ConversationBufferMemory

class SolutionAgent:
    def __init__(self, llama_llm, shared_memory, user_id, data_dir):  # Add user_id and data_dir
        self.llama_llm = llama_llm
        self.shared_memory = shared_memory
        self.user_id = user_id
        self.data_dir = data_dir
        self.file_path = os.path.join(data_dir, f"{self.user_id}_profile.json")

        # Load the solution prompt from file 
        self.solution_prompt_template = load_prompt_from_file("prompts/solution_prompt.txt")

    def run(self, user_message, analyzer_output, planner_output, shared_memory): # Add shared_memory
        """Generates a solution based on the analysis and plan."""

        # Access chat history and user profile 
        chat_history = shared_memory.load_memory_variables({}).get('history', [])  
        user_profile = self.load_user_profile()

        # Construct Llama's Prompt using .format() 
        llama_prompt = self.solution_prompt_template.format(
            user_message=user_message,
            analyzer_output=analyzer_output,
            planner_output=planner_output,
            chat_history=chat_history,
            user_profile=user_profile
        )

        try:
            llama_solution = self.llama_llm.invoke(llama_prompt).content
            print("Solution Agent:", llama_solution.strip())
            return llama_solution.strip()

        except Exception as e:
            print(f"Error in Solution Agent: {e}")
            return "Error in Solution Agent."

    def load_user_profile(self):
        """Loads the user's profile from the JSON file or creates a new one."""
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e: # Catch file and JSON errors (NEW)
            print(f"Error loading user profile: {e}")
            return {} # Return an empty dictionary if there's an error 