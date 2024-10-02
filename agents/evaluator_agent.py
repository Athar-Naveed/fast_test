import json
import os
from langchain_groq import ChatGroq
from utils import load_prompt_from_file
from langchain.memory import ConversationBufferMemory

class EvaluatorAgent:
    def __init__(self, llama_llm, shared_memory, user_id, data_dir):  # Add user_id and data_dir
        self.llama_llm = llama_llm
        self.shared_memory = shared_memory
        self.revision_attempts = 0
        self.user_id = user_id
        self.data_dir = data_dir
        self.file_path = os.path.join(data_dir, f"{self.user_id}_profile.json")

        # Load the evaluator prompt from a file (NEW)
        self.evaluator_prompt_template = load_prompt_from_file("prompts/evaluator_prompt.txt")

    def run(self, solution_output, shared_memory):  # Add shared_memory
        """Evaluates the proposed solution and decides if it needs revision."""

        # Access chat history and user profile 
        chat_history = shared_memory.load_memory_variables({}).get('history', [])  
        user_profile = self.load_user_profile()

        # Construct Llama's Prompt using .format() (NEW)
        llama_prompt = self.evaluator_prompt_template.format(
            solution_output=solution_output, 
            chat_history=chat_history,
            user_profile=user_profile 
        )

        try:
            llama_evaluation = self.llama_llm.invoke(llama_prompt).content
            print("Evaluator Agent:", llama_evaluation.strip())

            # Check if a revision was requested
            if "Yes, generate a revised solution" in llama_evaluation:
                self.revision_attempts += 1
                if self.revision_attempts < 3: # Limit to 3 revision attempts
                    print("Evaluator Agent: Requesting solution revision...\n")
                    return "REVISION_REQUESTED", llama_evaluation.strip() # Signal for revision
                else:
                    print("Evaluator Agent: Max revision attempts reached. Accepting current solution.\n")
                    return "ACCEPTED", llama_evaluation.strip() # Accept the solution

            else:
                self.revision_attempts = 0 # Reset the counter
                return "ACCEPTED", llama_evaluation.strip() # Accept the solution

        except Exception as e:
            print(f"Error in Evaluator Agent: {e}")
            return "ERROR", "Error in Evaluator Agent."

    def load_user_profile(self):
        """Loads the user's profile from the JSON file or creates a new one."""
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e: # Catch file and JSON errors
            print(f"Error loading user profile: {e}")
            return {} # Return an empty dictionary if there's an error 