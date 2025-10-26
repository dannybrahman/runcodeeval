"""
Metrics calculator for computing various evaluation scores with statistical confidence intervals
"""

from typing import Dict, List, Any, Tuple
from collections import defaultdict
import logging
import math


class MetricsCalculator:
    """Calculate evaluation metrics similar to runcodeeval with statistical confidence intervals"""
    
    # Z-values for common confidence levels
    Z_VALUES = {
        90: 1.645,
        95: 1.96,
        99: 2.576
    }
    
    def __init__(self, confidence_level: int = 95):
        """
        Initialize metrics calculator
        
        Args:
            confidence_level: Confidence level for intervals (90, 95, or 99)
        """
        self.logger = logging.getLogger(__name__)
        self.confidence_level = confidence_level
        if confidence_level not in self.Z_VALUES:
            raise ValueError(f"Confidence level must be one of {list(self.Z_VALUES.keys())}")
        self.z_value = self.Z_VALUES[confidence_level]
    
    def calculate_all_metrics(self, 
                            evaluation_results: List[Dict[str, Any]],
                            absent_solutions: List[Dict[str, Any]],
                            benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate all metrics including average score, by category, by complexity, and by object type"""
        
        total_benchmark_problems = len(benchmark_data)
        
        # Create a combined list including absent solutions with 0 score
        all_results = evaluation_results.copy()
        
        # Add absent solutions with 0 score (but NO error_type - they're just missing)
        for absent in absent_solutions:
            all_results.append({
                'task_id': absent['task_id'],
                'category': absent['category'],
                'complexity': absent['complexity'],
                'object_type': benchmark_data.get(absent['task_id'], {}).get('object', 'unknown'),
                'total_tests': 1,  # Consider it as having 1 test for scoring
                'tests_passed': 0,
                'all_tests_passed': False,
                'error_type': None,  # Absent solutions are not errors, just missing
                'error_message': 'Solution not provided (absent)'
            })
        
        # Verify we have all benchmark problems represented
        assert len(all_results) == total_benchmark_problems, \
            f"Mismatch: {len(all_results)} results vs {total_benchmark_problems} benchmark problems"
        
        # Calculate metrics
        metrics = {
            'average_score': self._calculate_average_score(all_results, benchmark_data),
            'score_by_category': self._calculate_score_by_category(all_results, benchmark_data),
            'score_by_complexity': self._calculate_score_by_complexity(all_results, benchmark_data),
            'score_by_object': self._calculate_score_by_object(all_results, benchmark_data),
            'total_problems': total_benchmark_problems,
            'solutions_provided': len(evaluation_results),
            'solutions_absent': len(absent_solutions),
            'overall_stats': self._calculate_overall_stats(evaluation_results),
            'confidence_level': self.confidence_level
        }
        
        return metrics
    
    def _calculate_confidence_interval(self, scores: List[float]) -> Dict[str, float]:
        """
        Calculate confidence interval for a list of scores
        
        Args:
            scores: List of individual problem scores (0-100)
            
        Returns:
            Dictionary with mean, std_dev, margin_of_error, lower_bound, upper_bound
        """
        if not scores:
            return {
                "mean": 0.0,
                "std_dev": 0.0,
                "margin_of_error": 0.0,
                "lower_bound": 0.0,
                "upper_bound": 0.0,
                "n": 0
            }
        
        n = len(scores)
        mean = sum(scores) / n
        
        # Calculate sample standard deviation
        if n > 1:
            variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0.0
        
        # Calculate margin of error: z * (s / sqrt(n))
        if n > 0 and std_dev > 0:
            margin_of_error = self.z_value * (std_dev / math.sqrt(n))
        else:
            margin_of_error = 0.0
        
        return {
            "mean": round(mean, 2),
            "std_dev": round(std_dev, 2), 
            "margin_of_error": round(margin_of_error, 2),
            "lower_bound": round(mean - margin_of_error, 2),
            "upper_bound": round(mean + margin_of_error, 2),
            "n": n
        }
    
    def _calculate_average_score(self, results: List[Dict[str, Any]], benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall average score across ALL benchmark problems with confidence interval
        
        Like a student taking an exam:
        - If student attempts 2 problems out of 10 and gets 1 correct
        - Score should be 1/10 = 10%, not 1/2 = 50%
        
        Returns:
            Dictionary with 'score' (mean) and 'confidence_interval' statistics
        """
        if not results or not benchmark_data:
            ci_stats = self._calculate_confidence_interval([])
            return {
                'score': 0.0,
                'confidence_interval': ci_stats
            }
        
        # Calculate individual problem scores (0-100 for each problem)
        problem_scores = []
        for result in results:
            if result['total_tests'] > 0:
                problem_score = (result['tests_passed'] / result['total_tests']) * 100
            else:
                problem_score = 0.0
            problem_scores.append(problem_score)
        
        # Calculate confidence interval
        ci_stats = self._calculate_confidence_interval(problem_scores)
        
        return {
            'score': ci_stats['mean'],  # The mean score (for backward compatibility)
            'confidence_interval': ci_stats
        }
    
    def _calculate_score_by_category(self, results: List[Dict[str, Any]], benchmark_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Calculate scores broken down by category with confidence intervals"""
        # First, count total problems per category in benchmark
        category_totals = defaultdict(int)
        for problem in benchmark_data.values():
            category = problem.get('topic', 'unknown')
            category_totals[category] += 1
        
        # Group results by category
        category_results = defaultdict(list)
        for result in results:
            category = result.get('category', 'unknown')
            category_results[category].append(result)
        
        # Calculate scores and confidence intervals for each category
        category_stats = {}
        for category, total_problems in category_totals.items():
            if category in category_results:
                # Get problem scores for this category
                problem_scores = []
                for result in category_results[category]:
                    if result['total_tests'] > 0:
                        problem_score = (result['tests_passed'] / result['total_tests']) * 100
                    else:
                        problem_score = 0.0
                    problem_scores.append(problem_score)
                
                # Add zeros for missing problems in this category
                missing_problems = total_problems - len(problem_scores)
                problem_scores.extend([0.0] * missing_problems)
            else:
                # No results for this category - all zeros
                problem_scores = [0.0] * total_problems
            
            # Calculate confidence interval
            ci_stats = self._calculate_confidence_interval(problem_scores)
            category_stats[category] = {
                'score': ci_stats['mean'],
                'confidence_interval': ci_stats
            }
        
        return dict(sorted(category_stats.items()))
    
    def _calculate_score_by_complexity(self, results: List[Dict[str, Any]], benchmark_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """Calculate scores grouped by complexity level with confidence intervals"""
        # First, count total problems per complexity in benchmark
        complexity_totals = defaultdict(int)
        for problem in benchmark_data.values():
            complexity = problem.get('complexity', 0)
            complexity_totals[complexity] += 1
        
        # Group results by complexity
        complexity_results = defaultdict(list)
        for result in results:
            complexity = result.get('complexity', 0)
            complexity_results[complexity].append(result)
        
        # Calculate scores and confidence intervals for each complexity level
        complexity_stats = {}
        for complexity in [1, 2, 3]:  # Standard complexity levels
            total_problems = complexity_totals.get(complexity, 0)
            
            if complexity in complexity_results and total_problems > 0:
                # Get problem scores for this complexity level
                problem_scores = []
                for result in complexity_results[complexity]:
                    if result['total_tests'] > 0:
                        problem_score = (result['tests_passed'] / result['total_tests']) * 100
                    else:
                        problem_score = 0.0
                    problem_scores.append(problem_score)
                
                # Add zeros for missing problems at this complexity level
                missing_problems = total_problems - len(problem_scores)
                problem_scores.extend([0.0] * missing_problems)
            else:
                # No results for this complexity level or no problems at this level
                problem_scores = [0.0] * total_problems if total_problems > 0 else []
            
            # Calculate confidence interval
            ci_stats = self._calculate_confidence_interval(problem_scores)
            complexity_stats[complexity] = {
                'score': ci_stats['mean'],
                'confidence_interval': ci_stats
            }
        
        return complexity_stats
    
    def _calculate_score_by_object(self, results: List[Dict[str, Any]], benchmark_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Calculate scores by object type (function vs class) with confidence intervals"""
        # First, count total problems per object type in benchmark
        object_totals = defaultdict(int)
        for problem in benchmark_data.values():
            object_type = problem.get('object', 'unknown')
            object_totals[object_type] += 1
        
        # Group results by object type
        object_results = defaultdict(list)
        for result in results:
            object_type = result.get('object_type', 'unknown')
            object_results[object_type].append(result)
        
        # Calculate scores and confidence intervals for each object type
        object_stats = {}
        for obj_type, total_problems in object_totals.items():
            if obj_type in object_results:
                # Get problem scores for this object type
                problem_scores = []
                for result in object_results[obj_type]:
                    if result['total_tests'] > 0:
                        problem_score = (result['tests_passed'] / result['total_tests']) * 100
                    else:
                        problem_score = 0.0
                    problem_scores.append(problem_score)
                
                # Add zeros for missing problems of this object type
                missing_problems = total_problems - len(problem_scores)
                problem_scores.extend([0.0] * missing_problems)
            else:
                # No results for this object type - all zeros
                problem_scores = [0.0] * total_problems
            
            # Calculate confidence interval
            ci_stats = self._calculate_confidence_interval(problem_scores)
            object_stats[obj_type] = {
                'score': ci_stats['mean'],
                'confidence_interval': ci_stats
            }
        
        return dict(sorted(object_stats.items()))
    
    def _calculate_overall_stats(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall statistics for provided solutions only"""
        if not evaluation_results:
            return {
                'total_evaluated': 0,
                'fully_correct': 0,
                'partially_correct': 0,
                'incorrect': 0,
                'syntax_errors': 0,
                'runtime_errors': 0,
                'timeout_errors': 0
            }
        
        stats = {
            'total_evaluated': len(evaluation_results),
            'fully_correct': sum(1 for r in evaluation_results 
                               if r['total_tests'] > 0 and r['tests_passed'] == r['total_tests']),
            'partially_correct': sum(1 for r in evaluation_results 
                                   if r['tests_passed'] > 0 and r['tests_passed'] < r['total_tests']),
            'incorrect': sum(1 for r in evaluation_results if r['tests_passed'] == 0),
            'syntax_errors': sum(1 for r in evaluation_results if r.get('error_type') == 'SyntaxError'),
            'runtime_errors': sum(1 for r in evaluation_results 
                                if r.get('error_type') in ['ExecutionError', 'NameError', 'TypeError']),
            'timeout_errors': sum(1 for r in evaluation_results if r.get('error_type') == 'TimeoutError')
        }
        
        return stats