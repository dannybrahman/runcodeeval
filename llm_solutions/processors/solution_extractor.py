"""Extract Python code from LLM responses using an agentic approach"""

import asyncio
import aiohttp
import json
from typing import Optional

class SolutionExtractor:
    """Extracts clean Python code from potentially noisy LLM responses using an LLM agent"""
    
    def __init__(self, extraction_api_key: str, extraction_model: str = "gpt-3.5-turbo"):
        """
        Initialize the solution extractor with an LLM for code extraction
        
        Args:
            extraction_api_key: API key for the extraction model (OpenAI)
            extraction_model: Model to use for extraction (fast/cheap model recommended)
        """
        self.extraction_api_key = extraction_api_key
        self.extraction_model = extraction_model
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
        # Extraction prompt template
        self.extraction_prompt = """You are a code extraction specialist. Your job is to extract only the executable Python code from the given text.

RULES:
1. Extract ONLY the Python code that solves the problem
2. Remove ALL explanations, comments, and markdown formatting
3. Remove text like "Here's the solution:", "```python", "```", etc.
4. Return ONLY the clean, executable Python code
5. If multiple code blocks exist, extract the main solution function/class
6. Preserve proper indentation and structure

TEXT TO EXTRACT FROM:
{response_text}

EXTRACTED CODE:"""

    async def extract_python_code(self, response: str, task_id: str = "") -> str:
        """
        Extract Python code from an LLM response using another LLM
        
        Args:
            response: Raw response text from LLM
            task_id: Optional task ID for logging
            
        Returns:
            Cleaned Python code string
        """
        if not response or not response.strip():
            return ""
        
        # First do a quick check - if response already looks clean, return as-is
        if self._looks_like_clean_code(response):
            return response.strip()
        
        # Use LLM agent to extract code
        try:
            extracted_code = await self._extract_with_agent(response)
            
            # Basic validation - ensure we got something that looks like code
            if extracted_code and self._contains_python_code(extracted_code):
                return extracted_code.strip()
            else:
                # Fallback: try basic cleanup if agent extraction fails
                return self._basic_cleanup(response)
                
        except Exception as e:
            print(f"Agent extraction failed for task {task_id}: {e}")
            # Fallback to basic cleanup
            return self._basic_cleanup(response)
    
    async def _extract_with_agent(self, response_text: str) -> str:
        """Use LLM agent to extract clean Python code"""
        headers = {
            "Authorization": f"Bearer {self.extraction_api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = self.extraction_prompt.format(response_text=response_text)
        
        payload = {
            "model": self.extraction_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.0,  # Deterministic extraction
            "max_tokens": 1000
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(self.base_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    raise Exception(f"Extraction API error: {response.status}")
    
    def _looks_like_clean_code(self, text: str) -> bool:
        """Check if text already looks like clean Python code"""
        text = text.strip()
        if not text:
            return False
        
        # Check for common signs that text is already clean
        has_markdown = "```" in text
        has_explanations = any(phrase in text.lower() for phrase in [
            "here's", "solution:", "answer:", "explanation:", "note:"
        ])
        
        # If no markdown or explanations and starts with def/class/import, likely clean
        first_line = text.split('\n')[0].strip()
        starts_with_code = any(first_line.startswith(keyword) for keyword in [
            "def ", "class ", "import ", "from "
        ])
        
        return starts_with_code and not has_markdown and not has_explanations
    
    def _contains_python_code(self, text: str) -> bool:
        """Check if extracted text contains valid Python code patterns"""
        if not text or not text.strip():
            return False
        
        # Look for Python keywords
        python_keywords = ['def ', 'class ', 'import ', 'from ', 'return ', 'if ', 'for ', 'while ']
        return any(keyword in text for keyword in python_keywords)
    
    def _basic_cleanup(self, response: str) -> str:
        """Fallback: basic regex-based cleanup if agent extraction fails"""
        import re
        
        if not response:
            return ""
        
        # Remove markdown code blocks
        pattern = r'```(?:python|py)?\s*\n?(.*?)\n?```'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            response = matches[0]
        
        # Remove common prefixes
        prefixes_to_remove = [
            r'^.*?here\'s.*?:?\s*\n?',
            r'^.*?solution.*?:?\s*\n?', 
            r'^.*?answer.*?:?\s*\n?',
            r'^.*?code.*?:?\s*\n?'
        ]
        
        for prefix in prefixes_to_remove:
            response = re.sub(prefix, '', response, flags=re.IGNORECASE | re.MULTILINE)
        
        return response.strip()

# Alternative implementation using different extraction models
class MultiModelSolutionExtractor(SolutionExtractor):
    """Solution extractor that can use different LLM providers for extraction"""
    
    def __init__(self, provider: str = "openai", **kwargs):
        """
        Initialize with different providers for extraction
        
        Args:
            provider: "openai", "anthropic", or "local" 
        """
        self.provider = provider
        if provider == "openai":
            super().__init__(**kwargs)
        # Could add other providers here for extraction