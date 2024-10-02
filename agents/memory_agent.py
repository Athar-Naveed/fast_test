import json 
import os
from langchain_groq import ChatGroq 
import re
from utils import load_prompt_from_file
from langchain.memory import ConversationBufferMemory

class MemoryAgent:
    def __init__(self, llama_llm, shared_memory, user_id="default_user", data_dir="user_data", file_path="default.json"):  # Add file_path
        self.llama_llm = llama_llm
        self.shared_memory = shared_memory
        self.user_id = user_id
        self.data_dir = data_dir
        self.file_path = file_path  # Store the file_path

        self.load_user_profile()  # Load the user profile when the agent is created

        # Load the memory agent prompt from a file (NEW)
        self.memory_prompt_template = load_prompt_from_file("prompts/memory_prompt.txt")

    def load_user_profile(self):
        """Loads the user's profile from the JSON file or creates a new one."""
        try:
            with open(self.file_path, "r") as f:
                self.user_profile = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e: # Catch file and JSON errors
            print(f"Error loading user profile: {e}")
            return {} # Return an empty dictionary if there's an error 

    def save_user_profile(self):
        """Saves the user profile to the JSON file."""
        with open(self.file_path, "w") as f:
            json.dump(self.user_profile, f, indent=4)
        print(f"User profile updated for {self.user_id}\n")

    def run(self, internal_dialogue):
        """Extracts SAVE_USER_INFO commands and updates the user profile."""

        # Construct the prompt for Llama to generate the updated JSON (Modified)
        llama_prompt = self.memory_prompt_template.format(
            internal_dialogue=internal_dialogue,
            user_profile=self.user_profile # Pass the dictionary directly
        )

        try:
            llama_json_output = self.llama_llm.invoke(llama_prompt).content

            print("**Llama's JSON Output:**") # Debugging output 
            print(llama_json_output)

            # Extract the JSON data from Llama's output
            json_pattern = r"```json\n(.*)\n```"
            json_match = re.search(json_pattern, llama_json_output, re.DOTALL)
            updated_json = json_match.group(1).strip() if json_match else None

            if updated_json:
                # Replace single quotes with double quotes for valid JSON
                updated_json = updated_json.replace("'", '"')
                try:
                    self.user_profile = json.loads(updated_json)
                    self.save_user_profile()
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
            else:
                print("Memory Agent: No JSON data found in Llama's output.")

        except Exception as e:
            print(f"Error in Memory Agent: {e}")