import os
from langchain.memory import ConversationBufferMemory

class InputAgent:
    def __init__(self, shared_memory, user_id, data_dir): # Add user_id and data_dir
        self.shared_memory = shared_memory
        self.user_id = user_id
        self.data_dir = data_dir
        self.file_path = os.path.join(data_dir, f"{self.user_id}_profile.json")

    def run(self, user_message):
        """Stores the user's message in shared memory."""
        self.shared_memory.save_context({"input": user_message}, {"output": "Waiting for chatbot response..."})  # Placeholder output
        print(f"Input Agent: User message stored: {user_message}")
        return "Analysis Agent, please analyze the user's message."