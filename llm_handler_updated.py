"""
LLM Handler for Navis Robot - Updated Version
Supports Groq (ultra-fast) and Hugging Face (free alternative)
Removed Perplexity (paid) and replaced with Hugging Face
"""

import requests
import json
from config import (
    GROQ_API_KEY, 
    HUGGINGFACE_API_KEY,
    PRIMARY_LLM,
    GROQ_MODEL,
    HUGGINGFACE_MODEL,
    MAX_RESPONSE_LENGTH,
    RESPONSE_TEMPERATURE,
    ROBOT_NAME,
    ROBOT_CREATOR
)


class LLMHandler:
    """
    Unified LLM handler supporting multiple FREE APIs
    - Groq: Ultra-fast inference (free, unlimited for now)
    - Hugging Face: Many models available (free tier ~1000 req/day)
    """
    
    def __init__(self):
        self.groq_key = GROQ_API_KEY
        self.hf_key = HUGGINGFACE_API_KEY
        self.primary = PRIMARY_LLM
        
        # API endpoints
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.hf_url = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL}"
        
        print(f"🧠 LLM Initialized: Primary={PRIMARY_LLM}, Model={GROQ_MODEL if PRIMARY_LLM == 'groq' else HUGGINGFACE_MODEL}")
    
    def ask(self, question, use_alternative=False):
        """
        Main method to get AI response
        
        Args:
            question (str): User's question
            use_alternative (bool): Force use of alternative LLM (Hugging Face)
        
        Returns:
            str: AI response
        """
        # If alternative is requested, use Hugging Face
        if use_alternative and self.hf_key != "YOUR_HF_API_KEY_HERE":
            return self._ask_huggingface(question)
        
        # Otherwise use primary LLM
        if self.primary == "huggingface" and self.hf_key != "YOUR_HF_API_KEY_HERE":
            return self._ask_huggingface(question)
        else:
            return self._ask_groq(question)
    
    def _ask_groq(self, question):
        """
        Query Groq API (ultra-fast, free)
        Free tier: Very generous limits, no credit card needed
        """
        if self.groq_key == "YOUR_GROQ_API_KEY_HERE":
            return "Please configure your Groq API key in config.py"
        
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = f"""You are {ROBOT_NAME}, a friendly humanoid robot built by {ROBOT_CREATOR}.
You are helpful, concise, and speak naturally like a person.
Keep responses brief (1-2 sentences) since you speak them out loud."""
        
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": RESPONSE_TEMPERATURE,
            "max_tokens": MAX_RESPONSE_LENGTH,
            "top_p": 1,
            "stream": False
        }
        
        try:
            response = requests.post(
                self.groq_url, 
                headers=headers, 
                json=payload, 
                timeout=15
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            
            print(f"✅ Groq Response: {answer[:50]}...")
            return answer
            
        except requests.exceptions.Timeout:
            return "Sorry, my brain is taking too long to respond."
        except requests.exceptions.RequestException as e:
            print(f"❌ Groq Error: {e}")
            # Fallback to Hugging Face if available
            if self.hf_key != "YOUR_HF_API_KEY_HERE":
                print("🔄 Falling back to Hugging Face...")
                return self._ask_huggingface(question)
            return "My AI brain is temporarily offline."
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            return "I encountered an error processing that."
    
    def _ask_huggingface(self, question):
        """
        Query Hugging Face Inference API (free alternative)
        Free tier: ~1000 requests/day
        """
        if self.hf_key == "YOUR_HF_API_KEY_HERE":
            return "Please configure your Hugging Face API key in config.py for alternative models."
        
        headers = {
            "Authorization": f"Bearer {self.hf_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = f"""You are {ROBOT_NAME}, a friendly humanoid robot built by {ROBOT_CREATOR}.
You are helpful, concise, and speak naturally like a person.
Keep responses brief (1-2 sentences) since you speak them out loud."""
        
        # Hugging Face format
        prompt = f"{system_prompt}\n\nUser: {question}\n{ROBOT_NAME}:"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": MAX_RESPONSE_LENGTH,
                "temperature": RESPONSE_TEMPERATURE,
                "top_p": 0.95,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(
                self.hf_url, 
                headers=headers, 
                json=payload, 
                timeout=20
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                answer = result[0].get('generated_text', '').strip()
            elif isinstance(result, dict):
                answer = result.get('generated_text', '').strip()
            else:
                answer = "I couldn't process that."
            
            print(f"✅ Hugging Face Response: {answer[:50]}...")
            return answer
            
        except requests.exceptions.Timeout:
            return "Sorry, my brain is taking too long to respond."
        except requests.exceptions.RequestException as e:
            print(f"❌ Hugging Face Error: {e}")
            # Fallback to Groq if available
            if self.groq_key != "YOUR_GROQ_API_KEY_HERE":
                print("🔄 Falling back to Groq...")
                return self._ask_groq(question)
            return "My AI brain is temporarily offline."
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            return "I encountered an error processing that."


# Singleton instance
_llm_instance = None

def get_llm():
    """Get or create LLM handler instance"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMHandler()
    return _llm_instance


# Convenience function
def ask_ai(question):
    """
    Simple function to ask AI a question
    """
    llm = get_llm()
    return llm.ask(question)


if __name__ == "__main__":
    # Test the LLM handler
    print("🧪 Testing LLM Handler...\n")
    
    llm = get_llm()
    
    # Test basic question
    print("Q: Who are you?")
    response = llm.ask("Who are you?")
    print(f"A: {response}\n")
    
    # Test another question
    print("Q: Tell me a joke")
    response = llm.ask("Tell me a joke")
    print(f"A: {response}\n")
