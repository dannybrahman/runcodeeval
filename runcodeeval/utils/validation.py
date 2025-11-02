"""
Validation utilities for solution and benchmark data formats
"""

from typing import Dict, Any
import logging


def validate_solution_format(solution: Dict[str, Any]) -> bool:
    """
    Validate that a solution object has the required keys
    
    Required keys: task_id, model, and either candidate_solution or extracted_solution
    Other keys are ignored
    """
    logger = logging.getLogger(__name__)
    
    # Check basic required keys
    basic_required = ['task_id', 'model']
    for key in basic_required:
        if key not in solution:
            logger.warning(f"Solution missing required key '{key}': {solution.get('task_id', 'unknown')}")
            return False
    
    # Check for solution code (either candidate_solution or extracted_solution)
    if 'candidate_solution' not in solution and 'extracted_solution' not in solution:
        logger.warning(f"Solution missing both 'candidate_solution' and 'extracted_solution': {solution.get('task_id', 'unknown')}")
        return False
    
    # Check for empty values
    if not solution['task_id'] or not solution['model']:
        logger.warning(f"Solution has empty required fields: {solution.get('task_id', 'unknown')}")
        return False
    
    # Solution code can be empty (will be treated as NoCompletionError)
    return True


def validate_benchmark_format(problem: Dict[str, Any]) -> bool:
    """
    Validate that a benchmark problem has the required structure
    
    Required keys: task_id, name, problem, canonical_solution, tests, topic, object, complexity
    """
    logger = logging.getLogger(__name__)
    
    required_keys = [
        'task_id', 'name', 'problem', 'canonical_solution', 
        'tests', 'topic', 'object', 'complexity'
    ]
    
    for key in required_keys:
        if key not in problem:
            logger.warning(f"Benchmark problem missing required key '{key}': {problem.get('task_id', 'unknown')}")
            return False
    
    # Check for empty values
    critical_keys = ['task_id', 'canonical_solution', 'tests']
    for key in critical_keys:
        if not problem[key]:
            logger.warning(f"Benchmark problem has empty critical field '{key}': {problem.get('task_id', 'unknown')}")
            return False
    
    # Validate object type
    if problem['object'] not in ['function', 'class', 'text']:
        logger.warning(f"Invalid object type '{problem['object']}': {problem.get('task_id', 'unknown')}")
        return False
    
    # Validate complexity
    if not isinstance(problem['complexity'], int) or problem['complexity'] not in [1, 2, 3]:
        logger.warning(f"Invalid complexity '{problem['complexity']}': {problem.get('task_id', 'unknown')}")
        return False
    
    return True