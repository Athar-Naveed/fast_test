import json
import os 
from utils import load_prompt_from_file
from langchain.memory import ConversationBufferMemory

class PlannerAgent:
    def __init__(self, llama_llm, shared_memory, user_id, data_dir):  # Add user_id and data_dir
        self.llama_llm = llama_llm
        self.shared_memory = shared_memory
        self.user_id = user_id
        self.data_dir = data_dir
        self.file_path = os.path.join(data_dir, f"{self.user_id}_profile.json")

        # Load the planner prompt from file
        self.planner_prompt_template = load_prompt_from_file("prompts/planner_prompt.txt") 

    def run(self, analyzer_output, shared_memory):  # Add shared_memory
        """Generates a plan based on the Analyzer Agent's output."""

        # Access chat history and user profile 
        chat_history = shared_memory.load_memory_variables({}).get('history', []) # Handle missing history 
        user_profile = self.load_user_profile()

        # Construct Llama's Prompt using .format()
        llama_prompt = self.planner_prompt_template.format(
            analyzer_output=analyzer_output,
            chat_history=chat_history,
            user_profile=user_profile  # Pass the dictionary directly 
        )

        # Generate Llama's Plan
        try:
            llama_plan = self.llama_llm.invoke(llama_prompt).content
            print("Planner Agent:", llama_plan.strip())  # Print for debugging
            return llama_plan.strip()

        except Exception as e:
            print(f"Error in Planner Agent: {e}")
            return "Error in Planner Agent."

    def load_user_profile(self):
        """Loads the user's profile from the JSON file or creates a new one."""
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e: # Catch file and JSON errors
            print(f"Error loading user profile: {e}")
            return {} # Return an empty dictionary if there's an error 