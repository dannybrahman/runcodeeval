"""
Error analyzer for categorizing and analyzing evaluation errors
"""

from typing import Dict, List, Any
from collections import defaultdict
import logging


class ErrorAnalyzer:
    """Analyze errors from evaluation results"""
    
    # Error categories matching runcodeeval
    # Note: AssertionError and TestFailure are NOT counted as errors
    # They represent solutions that run but produce incorrect results
    ERROR_CATEGORIES = {
        'NameError': 'NameError',
        'SyntaxError': 'SyntaxError', 
        'TimeoutError': 'TimeoutError',
        'NoCompletionError': 'NoCompletionError',
        'TypeError': 'Error',
        'ExecutionError': 'Error',
        'InspectionFailure': 'Error',
        'Error': 'Error'
        # Excluded: 'AssertionError', 'TestFailure' - these are scoring issues, not errors
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_errors(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze errors from evaluation results"""
        
        # Count errors by type
        error_counts = defaultdict(int)
        error_by_category = defaultdict(lambda: defaultdict(int))
        error_by_complexity = defaultdict(lambda: defaultdict(int))
        error_by_object = defaultdict(lambda: defaultdict(int))
        
        total_errors = 0
        error_examples = defaultdict(list)
        
        for result in evaluation_results:
            # Only process results that have an actual error_type (not None)
            # and are not just incorrect solutions
            if not result['all_tests_passed'] and result.get('error_type') is not None:
                error_type = result['error_type']
                # Only count errors that are actually in ERROR_CATEGORIES
                # AssertionError and TestFailure are not errors, just incorrect solutions
                # Absent solutions have error_type=None, so they won't be counted here
                if error_type in self.ERROR_CATEGORIES:
                    mapped_error = self.ERROR_CATEGORIES[error_type]
                    
                    error_counts[mapped_error] += 1
                    total_errors += 1
                    
                    # Track by category
                    category = result.get('category', 'unknown')
                    error_by_category[category][mapped_error] += 1
                    
                    # Track by complexity
                    complexity = result.get('complexity', 0)
                    error_by_complexity[complexity][mapped_error] += 1
                    
                    # Track by object type
                    object_type = result.get('object_type', 'unknown')
                    error_by_object[object_type][mapped_error] += 1
                    
                    # Collect examples (limit to 3 per error type)
                    if len(error_examples[mapped_error]) < 3:
                        error_examples[mapped_error].append({
                            'task_id': result['task_id'],
                            'message': result.get('error_message', 'No error message')
                        })
        
        # Calculate error rates
        total_evaluated = len(evaluation_results)
        error_rate = (total_errors / total_evaluated * 100) if total_evaluated > 0 else 0
        
        # Calculate error type percentages
        error_percentages = {}
        for error_type, count in error_counts.items():
            percentage = (count / total_errors * 100) if total_errors > 0 else 0
            error_percentages[error_type] = round(percentage, 2)
        
        # Build analysis result
        analysis = {
            'total_errors': total_errors,
            'error_rate': round(error_rate, 2),
            'error_counts': dict(error_counts),
            'error_percentages': error_percentages,
            'error_by_category': self._convert_nested_defaultdict(error_by_category),
            'error_by_complexity': self._convert_nested_defaultdict(error_by_complexity),
            'error_by_object': self._convert_nested_defaultdict(error_by_object),
            'error_examples': dict(error_examples)
        }
        
        # Add category-specific error rates
        category_error_rates = {}
        for category in error_by_category:
            category_total = sum(1 for r in evaluation_results if r.get('category') == category)
            category_errors = sum(error_by_category[category].values())
            if category_total > 0:
                category_error_rates[category] = round(category_errors / category_total * 100, 2)
        
        analysis['category_error_rates'] = category_error_rates
        
        # Add complexity-specific error rates
        complexity_error_rates = {}
        for complexity in [1, 2, 3]:
            complexity_total = sum(1 for r in evaluation_results if r.get('complexity') == complexity)
            complexity_errors = sum(error_by_complexity[complexity].values())
            if complexity_total > 0:
                complexity_error_rates[complexity] = round(complexity_errors / complexity_total * 100, 2)
        
        analysis['complexity_error_rates'] = complexity_error_rates
        
        return analysis
    
    def _convert_nested_defaultdict(self, nested_dict: defaultdict) -> Dict[str, Dict[str, int]]:
        """Convert nested defaultdict to regular dict for JSON serialization"""
        result = {}
        for key, inner_dict in nested_dict.items():
            result[key] = dict(inner_dict)
        return result