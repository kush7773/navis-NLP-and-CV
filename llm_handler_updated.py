"""
LLM Handler for Navis Robot - Updated Version
Supports Groq (ultra-fast) and Hugging Face (free alternative)
Includes training data injection for personal knowledge
"""

import requests
import json
import os
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

TRAINING_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data.json')


def load_training_data():
    """Load training data from JSON file"""
    try:
        if os.path.exists(TRAINING_DATA_PATH):
            with open(TRAINING_DATA_PATH, 'r') as f:
                data = json.load(f)
                return data.get('training_pairs', [])
    except Exception as e:
        print(f"⚠️ Could not load training data: {e}")
    return []


def save_training_data(pairs):
    """Save training data to JSON file"""
    try:
        with open(TRAINING_DATA_PATH, 'w') as f:
            json.dump({'training_pairs': pairs}, f, indent=4)
        return True
    except Exception as e:
        print(f"❌ Could not save training data: {e}")
        return False


def build_knowledge_context(pairs):
    """Build a knowledge context string from training pairs"""
    if not pairs:
        return ""
    
    context = "\n\nYou have been trained with the following knowledge. Use this to answer questions accurately:\n"
    for pair in pairs:
        context += f"\nQ: {pair['question']}\nA: {pair['answer']}\n"
    return context


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
        
        # Load training data
        self.training_pairs = load_training_data()
        self.knowledge_context = build_knowledge_context(self.training_pairs)
        
        print(f"🧠 LLM Initialized: Primary={PRIMARY_LLM}, Model={GROQ_MODEL if PRIMARY_LLM == 'groq' else HUGGINGFACE_MODEL}")
        print(f"📚 Training data loaded: {len(self.training_pairs)} knowledge pairs")
    
    def reload_training_data(self):
        """Reload training data from file"""
        self.training_pairs = load_training_data()
        self.knowledge_context = build_knowledge_context(self.training_pairs)
        print(f"🔄 Training data reloaded: {len(self.training_pairs)} pairs")
    
    def add_training_pair(self, question, answer):
        """Add a new Q&A training pair"""
        self.training_pairs.append({'question': question, 'answer': answer})
        save_training_data(self.training_pairs)
        self.knowledge_context = build_knowledge_context(self.training_pairs)
        print(f"✅ Added training pair: {question[:40]}...")
        return True
    
    def remove_training_pair(self, index):
        """Remove a training pair by index"""
        if 0 <= index < len(self.training_pairs):
            removed = self.training_pairs.pop(index)
            save_training_data(self.training_pairs)
            self.knowledge_context = build_knowledge_context(self.training_pairs)
            print(f"🗑️ Removed training pair: {removed['question'][:40]}...")
            return True
        return False
    
    def get_training_pairs(self):
        """Return all training pairs"""
        return self.training_pairs
    
    def _build_system_prompt(self):
        """Build system prompt with training data injected"""
        base_prompt = f"""You are {ROBOT_NAME}, a friendly humanoid robot built by {ROBOT_CREATOR} in collaboration with BNM Institute of Technology (BNMIT), Bangalore.
You were developed by Robomanthan, a robotics and AI company incubated at IIT Patna, founded by CEO Saurav Kumar and CTO Tanuj Kashyap.
You are helpful, concise, and speak naturally like a person.
Keep responses brief (1-3 sentences) since you speak them out loud.
If someone asks about Robomanthan, BNMIT, or about yourself, use the training knowledge provided below to answer accurately."""
        
        return base_prompt + self.knowledge_context
    
    def ask(self, question, use_alternative=False):
        """
        Main method to get AI response
        
        Args:
            question (str): User's question
            use_alternative (bool): Force use of alternative LLM (Hugging Face)
        
        Returns:
            str: AI response
        """
        # Check if question matches training data directly (fast local answer)
        local_answer = self._check_local_knowledge(question)
        if local_answer:
            return local_answer
        
        # If alternative is requested, use Hugging Face
        if use_alternative and self.hf_key != "YOUR_HF_API_KEY_HERE":
            return self._ask_huggingface(question)
        
        # Otherwise use primary LLM
        if self.primary == "huggingface" and self.hf_key != "YOUR_HF_API_KEY_HERE":
            return self._ask_huggingface(question)
        else:
            return self._ask_groq(question)
    
    def _check_local_knowledge(self, question):
        """Check if question closely matches any training pair"""
        question_lower = question.lower().strip()
        
        for pair in self.training_pairs:
            trained_q = pair['question'].lower().strip()
            # Direct match or very close
            if trained_q in question_lower or question_lower in trained_q:
                return pair['answer']
            
            # Check key phrase matching
            key_phrases = trained_q.replace('?', '').replace('what is ', '').replace('who is ', '').replace('tell me about ', '').split()
            if len(key_phrases) >= 2:
                match_count = sum(1 for kp in key_phrases if kp in question_lower)
                if match_count >= len(key_phrases) * 0.7:
                    return pair['answer']
        
        return None
    
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
        
        system_prompt = self._build_system_prompt()
        
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
        
        system_prompt = self._build_system_prompt()
        
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
    
    # Test training data
    print(f"Training pairs loaded: {len(llm.get_training_pairs())}")
    
    # Test basic question
    print("\nQ: Who are you?")
    response = llm.ask("Who are you?")
    print(f"A: {response}\n")
    
    # Test personal question
    print("Q: Who is the CEO of Robomanthan?")
    response = llm.ask("Who is the CEO of Robomanthan?")
    print(f"A: {response}\n")
    
    # Test BNMIT question
    print("Q: Who is the principal of BNMIT?")
    response = llm.ask("Who is the principal of BNMIT?")
    print(f"A: {response}\n")
