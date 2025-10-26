#!/usr/bin/env python3
"""
Test suite for MetricsCalculator to ensure exam-based scoring logic is correct
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator.metrics_calculator import MetricsCalculator


class TestMetricsCalculator(unittest.TestCase):
    """Test cases for MetricsCalculator exam-based scoring"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = MetricsCalculator()
    
    def _extract_score(self, score_data):
        """Helper to extract score from new CI structure"""
        if isinstance(score_data, dict) and 'score' in score_data:
            return score_data['score']
        return score_data
        
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = MetricsCalculator()
        
        self.benchmark_data = {
            'math_1': {'topic': 'math', 'complexity': 1, 'object': 'function'},
            'math_2': {'topic': 'math', 'complexity': 1, 'object': 'function'},
            'math_3': {'topic': 'math', 'complexity': 2, 'object': 'function'},
            'math_4': {'topic': 'math', 'complexity': 2, 'object': 'class'},
            'math_5': {'topic': 'math', 'complexity': 3, 'object': 'class'},
            'english_1': {'topic': 'english', 'complexity': 1, 'object': 'function'},
            'english_2': {'topic': 'english', 'complexity': 1, 'object': 'function'},
            'science_1': {'topic': 'science', 'complexity': 2, 'object': 'class'},
            'science_2': {'topic': 'science', 'complexity': 3, 'object': 'class'},
            'science_3': {'topic': 'science', 'complexity': 3, 'object': 'function'}
        }
        # Total: 10 problems
        # Math: 5 problems, English: 2 problems, Science: 3 problems
        # Level 1: 4 problems, Level 2: 3 problems, Level 3: 3 problems
        # Function: 6 problems, Class: 4 problems
    
    def test_perfect_score_all_attempted(self):
        """Test case: Student attempts all problems and gets all correct"""
        evaluation_results = [
            {'task_id': 'math_1', 'category': 'math', 'complexity': 1, 'object_type': 'function', 'total_tests': 3, 'tests_passed': 3},
            {'task_id': 'math_2', 'category': 'math', 'complexity': 1, 'object_type': 'function', 'total_tests': 2, 'tests_passed': 2},
            {'task_id': 'math_3', 'category': 'math', 'complexity': 2, 'object_type': 'function', 'total_tests': 1, 'tests_passed': 1},
            {'task_id': 'math_4', 'category': 'math', 'complexity': 2, 'object_type': 'class', 'total_tests': 4, 'tests_passed': 4},
            {'task_id': 'math_5', 'category': 'math', 'complexity': 3, 'object_type': 'class', 'total_tests': 2, 'tests_passed': 2},
            {'task_id': 'english_1', 'category': 'english', 'complexity': 1, 'object_type': 'function', 'total_tests': 1, 'tests_passed': 1},
            {'task_id': 'english_2', 'category': 'english', 'complexity': 1, 'object_type': 'function', 'total_tests': 3, 'tests_passed': 3},
            {'task_id': 'science_1', 'category': 'science', 'complexity': 2, 'object_type': 'class', 'total_tests': 2, 'tests_passed': 2},
            {'task_id': 'science_2', 'category': 'science', 'complexity': 3, 'object_type': 'class', 'total_tests': 1, 'tests_passed': 1},
            {'task_id': 'science_3', 'category': 'science', 'complexity': 3, 'object_type': 'function', 'total_tests': 2, 'tests_passed': 2}
        ]
        absent_solutions = []  # No absent solutions
        
        metrics = self.calculator.calculate_all_metrics(evaluation_results, absent_solutions, self.benchmark_data)
        
        # All problems attempted and correct = 100%
        self.assertEqual(self._extract_score(metrics['average_score']), 100.0)
        self.assertEqual(self._extract_score(metrics['score_by_category']['math']), 100.0)
        self.assertEqual(self._extract_score(metrics['score_by_category']['english']), 100.0)
        self.assertEqual(self._extract_score(metrics['score_by_category']['science']), 100.0)
        self.assertEqual(self._extract_score(metrics['score_by_complexity'][1]), 100.0)
        self.assertEqual(self._extract_score(metrics['score_by_complexity'][2]), 100.0)
        self.assertEqual(self._extract_score(metrics['score_by_complexity'][3]), 100.0)
        self.assertEqual(self._extract_score(metrics['score_by_object']['function']), 100.0)
        self.assertEqual(self._extract_score(metrics['score_by_object']['class']), 100.0)
    
    def test_partial_attempt_all_correct(self):
        """Test case: Student attempts only 2 out of 10 problems, gets both correct"""
        evaluation_results = [
            {'task_id': 'math_1', 'category': 'math', 'complexity': 1, 'object_type': 'function', 'total_tests': 3, 'tests_passed': 3},  # 100%
            {'task_id': 'science_2', 'category': 'science', 'complexity': 3, 'object_type': 'class', 'total_tests': 2, 'tests_passed': 2}   # 100%
        ]
        absent_solutions = [
            {'task_id': 'math_2', 'category': 'math', 'complexity': 1},
            {'task_id': 'math_3', 'category': 'math', 'complexity': 2},
            {'task_id': 'math_4', 'category': 'math', 'complexity': 2},
            {'task_id': 'math_5', 'category': 'math', 'complexity': 3},
            {'task_id': 'english_1', 'category': 'english', 'complexity': 1},
            {'task_id': 'english_2', 'category': 'english', 'complexity': 1},
            {'task_id': 'science_1', 'category': 'science', 'complexity': 2},
            {'task_id': 'science_3', 'category': 'science', 'complexity': 3}
        ]
        
        metrics = self.calculator.calculate_all_metrics(evaluation_results, absent_solutions, self.benchmark_data)
        
        # Total: (100 + 100 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0) / 10 = 20%
        self.assertEqual(self._extract_score(metrics['average_score']), 20.0)
        
        # Math: (100 + 0 + 0 + 0 + 0) / 5 = 20%
        self.assertEqual(self._extract_score(metrics['score_by_category']['math']), 20.0)
        
        # English: (0 + 0) / 2 = 0%
        self.assertEqual(self._extract_score(metrics['score_by_category']['english']), 0.0)
        
        # Science: (0 + 100 + 0) / 3 = 33.33%
        self.assertEqual(self._extract_score(metrics['score_by_category']['science']), 33.33)
        
        # Level 1: (100 + 0 + 0 + 0) / 4 = 25%
        self.assertEqual(self._extract_score(metrics['score_by_complexity'][1]), 25.0)
        
        # Level 2: (0 + 0 + 0) / 3 = 0%
        self.assertEqual(self._extract_score(metrics['score_by_complexity'][2]), 0.0)
        
        # Level 3: (0 + 100 + 0) / 3 = 33.33%
        self.assertEqual(self._extract_score(metrics['score_by_complexity'][3]), 33.33)
        
        # Function: (100 + 0 + 0 + 0 + 0 + 0) / 6 = 16.67%
        self.assertEqual(self._extract_score(metrics['score_by_object']['function']), 16.67)
        
        # Class: (0 + 0 + 100 + 0) / 4 = 25%
        self.assertEqual(self._extract_score(metrics['score_by_object']['class']), 25.0)
    
    def test_partial_attempt_partial_correct(self):
        """Test case: Student attempts 3 problems, gets 1 fully correct, 1 partially correct, 1 wrong"""
        evaluation_results = [
            {'task_id': 'math_1', 'category': 'math', 'complexity': 1, 'object_type': 'function', 'total_tests': 4, 'tests_passed': 4},  # 100%
            {'task_id': 'math_2', 'category': 'math', 'complexity': 1, 'object_type': 'function', 'total_tests': 2, 'tests_passed': 1},  # 50%
            {'task_id': 'english_1', 'category': 'english', 'complexity': 1, 'object_type': 'function', 'total_tests': 3, 'tests_passed': 0}   # 0%
        ]
        absent_solutions = [
            {'task_id': 'math_3', 'category': 'math', 'complexity': 2},
            {'task_id': 'math_4', 'category': 'math', 'complexity': 2},
            {'task_id': 'math_5', 'category': 'math', 'complexity': 3},
            {'task_id': 'english_2', 'category': 'english', 'complexity': 1},
            {'task_id': 'science_1', 'category': 'science', 'complexity': 2},
            {'task_id': 'science_2', 'category': 'science', 'complexity': 3},
            {'task_id': 'science_3', 'category': 'science', 'complexity': 3}
        ]
        
        metrics = self.calculator.calculate_all_metrics(evaluation_results, absent_solutions, self.benchmark_data)
        
        # Total: (100 + 50 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0) / 10 = 15%
        self.assertEqual(self._extract_score(metrics['average_score']), 15.0)
        
        # Math: (100 + 50 + 0 + 0 + 0) / 5 = 30%
        self.assertEqual(self._extract_score(metrics['score_by_category']['math']), 30.0)
        
        # English: (0 + 0) / 2 = 0%
        self.assertEqual(self._extract_score(metrics['score_by_category']['english']), 0.0)
        
        # Science: (0 + 0 + 0) / 3 = 0%
        self.assertEqual(self._extract_score(metrics['score_by_category']['science']), 0.0)
        
        # Level 1: (100 + 50 + 0 + 0) / 4 = 37.5%
        self.assertEqual(self._extract_score(metrics['score_by_complexity'][1]), 37.5)
        
        # Function: (100 + 50 + 0 + 0 + 0 + 0) / 6 = 25%
        self.assertEqual(self._extract_score(metrics['score_by_object']['function']), 25.0)
        
        # Class: (0 + 0 + 0 + 0) / 4 = 0%
        self.assertEqual(self._extract_score(metrics['score_by_object']['class']), 0.0)
    
    def test_no_attempts(self):
        """Test case: Student attempts no problems (all absent)"""
        evaluation_results = []
        absent_solutions = [
            {'task_id': task_id, 'category': data['topic'], 'complexity': data['complexity']}
            for task_id, data in self.benchmark_data.items()
        ]
        
        metrics = self.calculator.calculate_all_metrics(evaluation_results, absent_solutions, self.benchmark_data)
        
        # All scores should be 0%
        self.assertEqual(self._extract_score(metrics['average_score']), 0.0)
        self.assertEqual(self._extract_score(metrics['score_by_category']['math']), 0.0)
        self.assertEqual(self._extract_score(metrics['score_by_category']['english']), 0.0)
        self.assertEqual(self._extract_score(metrics['score_by_category']['science']), 0.0)
        self.assertEqual(self._extract_score(metrics['score_by_complexity'][1]), 0.0)
        self.assertEqual(self._extract_score(metrics['score_by_complexity'][2]), 0.0)
        self.assertEqual(self._extract_score(metrics['score_by_complexity'][3]), 0.0)
        self.assertEqual(self._extract_score(metrics['score_by_object']['function']), 0.0)
        self.assertEqual(self._extract_score(metrics['score_by_object']['class']), 0.0)
    
    def test_summary_statistics(self):
        """Test that summary statistics are correct"""
        evaluation_results = [
            {'task_id': 'math_1', 'category': 'math', 'complexity': 1, 'object_type': 'function', 'total_tests': 2, 'tests_passed': 2}
        ]
        absent_solutions = [
            {'task_id': task_id, 'category': data['topic'], 'complexity': data['complexity']}
            for task_id, data in self.benchmark_data.items() if task_id != 'math_1'
        ]
        
        metrics = self.calculator.calculate_all_metrics(evaluation_results, absent_solutions, self.benchmark_data)
        
        # Check summary counts
        self.assertEqual(metrics['total_problems'], 10)  # Total benchmark problems
        self.assertEqual(metrics['solutions_provided'], 1)  # Only 1 solution provided
        self.assertEqual(metrics['solutions_absent'], 9)   # 9 solutions absent
    
    def test_edge_case_zero_tests(self):
        """Test edge case where a solution has 0 total tests"""
        evaluation_results = [
            {'task_id': 'math_1', 'category': 'math', 'complexity': 1, 'object_type': 'function', 'total_tests': 0, 'tests_passed': 0}  # 0/0 = 0%
        ]
        absent_solutions = [
            {'task_id': task_id, 'category': data['topic'], 'complexity': data['complexity']}
            for task_id, data in self.benchmark_data.items() if task_id != 'math_1'
        ]
        
        metrics = self.calculator.calculate_all_metrics(evaluation_results, absent_solutions, self.benchmark_data)
        
        # Should handle 0/0 as 0% gracefully
        self.assertEqual(self._extract_score(metrics['average_score']), 0.0)
        self.assertEqual(self._extract_score(metrics['score_by_category']['math']), 0.0)


def run_metrics_tests():
    """Convenience function to run all metrics tests"""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_metrics_tests()