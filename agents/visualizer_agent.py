import json
from utils import load_prompt_from_file
import re
class VisualizerAgent:
    def __init__(self, llama_llm, shared_memory, user_id, data_dir):
        self.llama_llm = llama_llm
        self.shared_memory = shared_memory
        self.user_id = user_id
        self.data_dir = data_dir
        self.visualizer_prompt_template = load_prompt_from_file("prompts/visualizer_prompt.txt")

    def run(self, user_message, internal_dialogue, shared_memory):
        """Determines if a visualization is needed and generates Mermaid code."""

        chat_history = shared_memory.load_memory_variables({}).get('history', [])
        user_profile = self.load_user_profile()

        llama_prompt = self.visualizer_prompt_template.format(
            user_message=user_message,
            internal_dialogue=internal_dialogue,
            chat_history=chat_history,
            user_profile=user_profile # No need for json.dumps
        )

        max_attempts = 3
        attempts = 0

        while attempts < max_attempts:
            try:
                llama_output = self.llama_llm.invoke(llama_prompt).content
                print("Visualizer Agent Output:", llama_output.strip())

                # Check if visualization is needed
                if "No visualization needed." in llama_output:
                    return False, ""
                
                # Extract Manim code (handling missing code block)
                code_match = re.search(r"```python(.*?)```", llama_output, re.DOTALL)
                if code_match:
                    manim_code = code_match.group(1).strip()
                else:
                    raise ValueError("Invalid Manim code format. Missing code block or no code generated.") # Raise a more descriptive error


                # Prepare the Manim code
                prepared_code = self.prepare_manim_code(manim_code)

                # --- Refinement Prompt for Llama ---
                refinement_prompt = f"""
                You are PathAI's Visualization Agent. Your task is to refine and improve the given Manim code to create a more effective and engaging animation.

                Here's the Manim code to refine:
                ```python
                {prepared_code}
                ```

                ## Instructions for Refinement:

                * **Clarity and Detail:** Ensure animation clearly illustrates concepts or steps. Add details, explanations, or visual elements.
                * **Visual Appeal:** Use appropriate colors, fonts, animations, and transitions. Pay attention to placement and timing of elements.
                * **Accuracy:** Double-check visualization accurately represents information/concepts.
                * **Conciseness:** Remove unnecessary/redundant code/visual elements.
                * **User Interest:** Tailor the visualization to the user's interests/learning style, if possible, based on profile information.

                Refined Manim code (enclosed in triple backticks and syntactically correct):
                ```python
                """
                try:
                    refinement_output = self.llama_llm.invoke(refinement_prompt).content  # Get refinement output
                    
                    # Extract refined Manim code (handling missing code block)
                    refined_code_match = re.search(r"```python(.*?)```", refinement_output, re.DOTALL)
                    if refined_code_match:
                      refined_manim_code = refined_code_match.group(1).strip()

                      # Update llama_prompt for next iteration (if needed)
                      llama_prompt = refinement_prompt.replace(f"```python\n{prepared_code}\n```", f"```python\n{refined_manim_code}\n```")  # Precise replacement
                       # Prepare the refined Manim code
                      prepared_code = self.prepare_manim_code(refined_manim_code)
                    
                    elif "No visualization needed." in refinement_output:
                        return False, ""
                    else:
                        raise ValueError("Invalid Manim code format during refinement. Missing code block or no code generated.")

                except Exception as e:
                    print(f"Error during Manim code refinement: {e}")
                    return False, ""
                
            except Exception as e:  # {{ edit_1 }}
                print("Error occurred:", e)

            attempts += 1

            return True, prepared_code


    def load_user_profile(self):
        """Loads the user's profile from the JSON file or creates a new one."""
        file_path = f"{self.data_dir}/{self.user_id}_profile.json"
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading user profile: {e}")
            return {}

    def prepare_manim_code(self, code):
        """Prepares the Manim code by adding necessary imports and class structure."""
        prepared_code = f"""
{code}
"""
        return prepared_code