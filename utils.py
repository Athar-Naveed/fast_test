import os
import json
import re

def save_conversation_history(user_id, data_dir, shared_memory):
  """Saves the conversation history to a JSON file."""
  file_path = os.path.join(data_dir, f"{user_id}_chat_history.json")
  chat_history = shared_memory.load_memory_variables({})['history']

  with open(file_path, "w") as f:
    json.dump(chat_history, f, indent=4)

  print(f"Conversation history saved for {user_id}\n")

def load_conversation_history(user_id, data_dir, shared_memory):
    """Loads conversation history from a JSON file."""
    file_path = os.path.join(data_dir, f"{user_id}_chat_history.json")
    try:
        with open(file_path, "r") as f:
            chat_history = json.load(f)
        
        # Split the chat history into turns
        turns = re.split(r'(Human:|AI:)', chat_history)[1:]  # [1:] to remove the empty string at the start
        
        # Pair up the labels with their content
        turns = [(turns[i], turns[i+1].strip()) for i in range(0, len(turns), 2)]
        
        # Clear existing messages in shared memory
        shared_memory.chat_memory.clear()
        
        # Iterate through the turns and save to shared memory
        for label, content in turns:
            if label == "Human:":
                shared_memory.chat_memory.add_user_message(content)
            elif label == "AI:":
                shared_memory.chat_memory.add_ai_message(content)

        return True, f"Conversation history loaded for {user_id}"
    except FileNotFoundError:
        return False, f"No conversation history found for {user_id}."
    except json.JSONDecodeError:
        return False, f"Error decoding conversation history for {user_id}."

def load_prompt_from_file(file_path):
    """Loads a prompt from a text file."""
    with open(file_path, "r") as f:
        return f.read().strip()