"""Standardized prompt formatting for consistent LLM interactions"""

from typing import Dict

class PromptTemplate:
    """Handles standardized prompt formatting across different LLM providers"""
    
    @staticmethod
    def format_problem_prompt(problem_data: Dict) -> str:
        """
        Format a problem into a standardized prompt
        
        Args:
            problem_data: Dict containing 'problem', 'name', and other metadata
            
        Returns:
            Formatted prompt string
        """
        problem_description = problem_data["problem"]
        function_name = problem_data["name"]
        
        prompt = f"""Problem: {problem_description}

Requirements:
- Function/class name must be exactly: {function_name}
- Return only Python code
- No explanations or comments
- No markdown formatting
- No additional text"""
        
        return prompt
    
    @staticmethod
    def get_system_prompt() -> str:
        """Get the standard system prompt for code generation"""
        return "You are a Python programming expert. You will be given programming problems and must respond with ONLY the Python code solution. Do not include any explanations, comments, or markdown formatting. Return only the executable Python code."
    
    @staticmethod
    def format_for_chat_model(problem_data: Dict) -> Dict:
        """
        Format for chat-based models (OpenAI, Anthropic, etc.)
        
        Returns:
            Dict with 'system' and 'user' messages
        """
        return {
            "system": PromptTemplate.get_system_prompt(),
            "user": PromptTemplate.format_problem_prompt(problem_data)
        }
    
    @staticmethod
    def format_for_completion_model(problem_data: Dict) -> str:
        """
        Format for completion-based models (some Hugging Face models)
        
        Returns:
            Single prompt string with system and user content combined
        """
        system_prompt = PromptTemplate.get_system_prompt()
        user_prompt = PromptTemplate.format_problem_prompt(problem_data)
        
        return f"{system_prompt}\n\n{user_prompt}\n\nSolution:"