# api_handler.py
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DeepseekAPI:
    def __init__(self):
        self.base_url = "https://api.deepseek.com/v1"
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in .env file")
        
    def query(self, prompt, context, web_search=True, r1_reasoning=False):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "prompt": f"Context:\n{context}\n\nQuestion: {prompt}",
            "web_search": web_search,
            "reasoning_mode": "r1" if r1_reasoning else "basic",
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            return f"API Error: {str(e)}"
        except KeyError:
            return "Error: Invalid API response format"