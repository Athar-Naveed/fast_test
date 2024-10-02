import json
import os
import re 
from textblob import TextBlob
from langchain_groq import ChatGroq
from utils import load_prompt_from_file
from langchain.memory import ConversationBufferMemory

class AnalyzerAgent:
    def __init__(self, llama_llm, shared_memory, user_id, data_dir):  # Add user_id and data_dir
        self.llama_llm = llama_llm
        self.shared_memory = shared_memory
        self.user_id = user_id
        self.data_dir = data_dir
        self.file_path = os.path.join(data_dir, f"{self.user_id}_profile.json")

        # Load the analyzer prompt 
        self.analyzer_prompt_template = load_prompt_from_file("prompts/analyzer_prompt.txt") 

    def run(self, user_message, shared_memory):  # Add shared_memory parameter
        """Analyzes the user's message using Llama 3.1."""

        # Access chat history and user profile (new)
        chat_history = shared_memory.load_memory_variables({}).get('history', []) 
        # .get('history', []) will return an empty list if 'history' is not found
        user_profile = self.load_user_profile()

        # Construct Llama's Prompt
        llama_prompt = self.analyzer_prompt_template.format(
            user_message=user_message,
            chat_history=chat_history,
            user_profile=json.dumps(user_profile, indent=4)
        )

        # Generate Llama's Analysis (Corrected Call)
        try:
            llama_analysis = self.llama_llm.invoke(llama_prompt).content  # Use .invoke().content
            print("Analyzer Agent:", llama_analysis.strip()) # Print the analysis for debugging
            return llama_analysis.strip()

        except Exception as e:
            print(f"Error in Analyzer Agent: {e}")
            return "Error in Analyzer Agent."


    def load_user_profile(self):
        """Loads the user's profile from the JSON file or creates a new one."""
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e: # Catch file and JSON errors
            print(f"Error loading user profile: {e}")
            return {} # Return an empty dictionary if there's an error 