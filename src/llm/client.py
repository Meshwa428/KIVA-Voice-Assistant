from openai import OpenAI
from typing import List, Dict, Generator, Union

class OllamaClient:
    def __init__(self, model_name: str, system_prompt: str, base_url: str = "http://localhost:11434/v1", llm_options: dict = None):
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama", # Required but unused by Ollama
        )
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.llm_options = llm_options or {}
        self.history: List[Dict[str, str]] = [
            {'role': 'system', 'content': self.system_prompt}
        ]

    def chat(self, user_text: str, stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """
        Sends user text to the LLM and returns the response.
        Updates internal history.
        If stream=True, returns a generator yielding response chunks.
        """
        self.history.append({'role': 'user', 'content': user_text})
        
        # 1. Separate Standard OpenAI params from Ollama-specific options
        openai_params = {}
        ollama_options = self.llm_options.copy()
        
        # Map specific keys that OpenAI client supports natively
        if "temperature" in ollama_options:
            openai_params["temperature"] = ollama_options.pop("temperature")
        if "top_p" in ollama_options:
            openai_params["top_p"] = ollama_options.pop("top_p")
        if "seed" in ollama_options:
            openai_params["seed"] = ollama_options.pop("seed")
        if "stop" in ollama_options:
            openai_params["stop"] = ollama_options.pop("stop")
        if "num_predict" in ollama_options:
            openai_params["max_tokens"] = ollama_options.pop("num_predict")
            
        # 2. Prepare extra_body
        # Default to 4 threads if not specified to prevent CPU starvation
        final_ollama_options = ollama_options
        if "num_thread" not in final_ollama_options:
            final_ollama_options["num_thread"] = 4
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history[-21:], # Keep context window reasonable
                stream=stream,
                extra_body={"options": final_ollama_options},
                **openai_params
            )
            
            if stream:
                return self._stream_response(response)
            else:
                assistant_response = response.choices[0].message.content
                self.history.append({'role': 'assistant', 'content': assistant_response})
                return assistant_response
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return "Oops, my brain short-circuited for a second! Can you say that again?"

    def _stream_response(self, response_generator) -> Generator[str, None, None]:
        """Helper to yield chunks and accumulate the full response for history."""
        full_response = ""
        for chunk in response_generator:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
                yield content
        
        self.history.append({'role': 'assistant', 'content': full_response})

    def clear_history(self):
        """Resets conversation history to just the system prompt."""
        self.history = [{'role': 'system', 'content': self.system_prompt}]
