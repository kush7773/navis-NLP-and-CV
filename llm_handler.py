"""
LLM Handler for Navis Robot
Supports multiple free LLM APIs with fallback options
"""

import requests
import json
from config import (
    GROQ_API_KEY, 
    PERPLEXITY_API_KEY,
    PRIMARY_LLM,
    GROQ_MODEL,
    PERPLEXITY_MODEL,
    MAX_RESPONSE_LENGTH,
    RESPONSE_TEMPERATURE,
    ROBOT_NAME,
    ROBOT_CREATOR
)


class LLMHandler:
    """
    Unified LLM handler supporting multiple APIs
    - Groq: Ultra-fast inference (free)
    - Perplexity: Internet-connected responses (free tier)
    """
    
    def __init__(self):
        self.groq_key = GROQ_API_KEY
        self.perplexity_key = PERPLEXITY_API_KEY
        self.primary = PRIMARY_LLM
        
        # API endpoints
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        self.perplexity_url = "https://api.perplexity.ai/chat/completions"
        
        print(f"🧠 LLM Initialized: Primary={PRIMARY_LLM}, Model={GROQ_MODEL if PRIMARY_LLM == 'groq' else PERPLEXITY_MODEL}")
    
    def ask(self, question, use_internet=False):
        """
        Main method to get AI response
        
        Args:
            question (str): User's question
            use_internet (bool): Force use of internet-connected LLM (Perplexity)
        
        Returns:
            str: AI response
        """
        # If internet access is needed, use Perplexity
        if use_internet and self.perplexity_key != "YOUR_PERPLEXITY_API_KEY_HERE":
            return self._ask_perplexity(question)
        
        # Otherwise use primary LLM
        if self.primary == "perplexity" and self.perplexity_key != "YOUR_PERPLEXITY_API_KEY_HERE":
            return self._ask_perplexity(question)
        else:
            return self._ask_groq(question)
    
    def _ask_groq(self, question):
        """
        Query Groq API (ultra-fast, no internet access)
        Free tier: Very generous limits
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
            # Fallback to Perplexity if available
            if self.perplexity_key != "YOUR_PERPLEXITY_API_KEY_HERE":
                print("🔄 Falling back to Perplexity...")
                return self._ask_perplexity(question)
            return "My AI brain is temporarily offline."
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            return "I encountered an error processing that."
    
    def _ask_perplexity(self, question):
        """
        Query Perplexity API (has internet access for current information)
        Free tier: Limited but available
        """
        if self.perplexity_key == "YOUR_PERPLEXITY_API_KEY_HERE":
            return "Please configure your Perplexity API key in config.py for internet-connected responses."
        
        headers = {
            "Authorization": f"Bearer {self.perplexity_key}",
            "Content-Type": "application/json"
        }
        
        system_prompt = f"""You are {ROBOT_NAME}, a friendly humanoid robot built by {ROBOT_CREATOR}.
You have access to the internet and can provide current information.
Keep responses brief (1-2 sentences) since you speak them out loud."""
        
        payload = {
            "model": PERPLEXITY_MODEL,
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
                self.perplexity_url, 
                headers=headers, 
                json=payload, 
                timeout=15
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result['choices'][0]['message']['content'].strip()
            
            print(f"✅ Perplexity Response (Internet): {answer[:50]}...")
            return answer
            
        except requests.exceptions.Timeout:
            return "Sorry, my brain is taking too long to respond."
        except requests.exceptions.RequestException as e:
            print(f"❌ Perplexity Error: {e}")
            # Fallback to Groq if available
            if self.groq_key != "YOUR_GROQ_API_KEY_HERE":
                print("🔄 Falling back to Groq...")
                return self._ask_groq(question)
            return "My AI brain is temporarily offline."
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            return "I encountered an error processing that."
    
    def check_internet_keywords(self, question):
        """
        Detect if question requires internet access
        Returns True if question likely needs current information
        """
        internet_keywords = [
            'weather', 'news', 'current', 'today', 'now', 'latest',
            'stock', 'price', 'score', 'happening', 'update',
            'who is', 'what is happening', 'recent', 'this week'
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in internet_keywords)


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
    Automatically detects if internet access is needed
    """
    llm = get_llm()
    
    # Check if question needs internet
    needs_internet = llm.check_internet_keywords(question)
    
    return llm.ask(question, use_internet=needs_internet)


if __name__ == "__main__":
    # Test the LLM handler
    print("🧪 Testing LLM Handler...\n")
    
    llm = get_llm()
    
    # Test basic question
    print("Q: Who are you?")
    response = llm.ask("Who are you?")
    print(f"A: {response}\n")
    
    # Test internet-connected question (if Perplexity is configured)
    print("Q: What's the weather like today?")
    response = llm.ask("What's the weather like today?", use_internet=True)
    print(f"A: {response}\n")
