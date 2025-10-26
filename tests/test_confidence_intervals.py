#!/usr/bin/env python3
"""
Test suite for confidence interval functionality in MetricsCalculator
"""

import unittest
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator.metrics_calculator import MetricsCalculator


class TestConfidenceIntervals(unittest.TestCase):
    """Test cases for confidence interval calculations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = MetricsCalculator(confidence_level=95)
        
        # Sample benchmark data
        self.benchmark_data = {
            'math_1': {'topic': 'math', 'complexity': 1, 'object': 'function'},
            'math_2': {'topic': 'math', 'complexity': 1, 'object': 'function'},
            'math_3': {'topic': 'math', 'complexity': 2, 'object': 'function'},
            'english_1': {'topic': 'english', 'complexity': 1, 'object': 'function'},
            'english_2': {'topic': 'english', 'complexity': 1, 'object': 'function'}
        }
    
    def test_confidence_interval_structure(self):
        """Test that confidence intervals are properly structured"""
        evaluation_results = [
            {'task_id': 'math_1', 'category': 'math', 'complexity': 1, 'object_type': 'function', 'total_tests': 4, 'tests_passed': 4},
            {'task_id': 'math_2', 'category': 'math', 'complexity': 1, 'object_type': 'function', 'total_tests': 2, 'tests_passed': 1}
        ]
        absent_solutions = [
            {'task_id': 'math_3', 'category': 'math', 'complexity': 2},
            {'task_id': 'english_1', 'category': 'english', 'complexity': 1},
            {'task_id': 'english_2', 'category': 'english', 'complexity': 1}
        ]
        
        metrics = self.calculator.calculate_all_metrics(evaluation_results, absent_solutions, self.benchmark_data)
        
        # Check that all metrics now return dictionaries with confidence intervals
        self.assertIsInstance(metrics['average_score'], dict)
        self.assertIn('score', metrics['average_score'])
        self.assertIn('confidence_interval', metrics['average_score'])
        
        # Check confidence interval structure
        ci = metrics['average_score']['confidence_interval']
        self.assertIn('mean', ci)
        self.assertIn('std_dev', ci)
        self.assertIn('margin_of_error', ci)
        self.assertIn('lower_bound', ci)
        self.assertIn('upper_bound', ci)
        self.assertIn('n', ci)
        
        # Check that n equals total benchmark problems
        self.assertEqual(ci['n'], 5)
        
        # Check confidence level is stored 
        self.assertEqual(metrics['confidence_level'], 95)
    
    def test_category_confidence_intervals(self):
        """Test that category scores have confidence intervals"""
        evaluation_results = [
            {'task_id': 'math_1', 'category': 'math', 'complexity': 1, 'object_type': 'function', 'total_tests': 2, 'tests_passed': 2}
        ]
        absent_solutions = [
            {'task_id': 'math_2', 'category': 'math', 'complexity': 1},
            {'task_id': 'math_3', 'category': 'math', 'complexity': 2},
            {'task_id': 'english_1', 'category': 'english', 'complexity': 1},
            {'task_id': 'english_2', 'category': 'english', 'complexity': 1}
        ]
        
        metrics = self.calculator.calculate_all_metrics(evaluation_results, absent_solutions, self.benchmark_data)
        
        # Check category scores have confidence intervals
        for category, score_data in metrics['score_by_category'].items():
            self.assertIsInstance(score_data, dict)
            self.assertIn('score', score_data)
            self.assertIn('confidence_interval', score_data)
    
    def test_perfect_score_low_variance(self):
        """Test that perfect scores have low variance/margin of error"""
        evaluation_results = [
            {'task_id': 'math_1', 'category': 'math', 'complexity': 1, 'object_type': 'function', 'total_tests': 3, 'tests_passed': 3},
            {'task_id': 'math_2', 'category': 'math', 'complexity': 1, 'object_type': 'function', 'total_tests': 2, 'tests_passed': 2},
            {'task_id': 'math_3', 'category': 'math', 'complexity': 2, 'object_type': 'function', 'total_tests': 1, 'tests_passed': 1},
            {'task_id': 'english_1', 'category': 'english', 'complexity': 1, 'object_type': 'function', 'total_tests': 4, 'tests_passed': 4},
            {'task_id': 'english_2', 'category': 'english', 'complexity': 1, 'object_type': 'function', 'total_tests': 2, 'tests_passed': 2}
        ]
        absent_solutions = []
        
        metrics = self.calculator.calculate_all_metrics(evaluation_results, absent_solutions, self.benchmark_data)
        
        # Perfect score should have 0 standard deviation and margin of error
        ci = metrics['average_score']['confidence_interval']
        self.assertEqual(ci['mean'], 100.0)
        self.assertEqual(ci['std_dev'], 0.0)
        self.assertEqual(ci['margin_of_error'], 0.0)
        self.assertEqual(ci['lower_bound'], 100.0)
        self.assertEqual(ci['upper_bound'], 100.0)
    
    def test_mixed_scores_variance(self):
        """Test that mixed scores have meaningful variance"""
        evaluation_results = [
            {'task_id': 'math_1', 'category': 'math', 'complexity': 1, 'object_type': 'function', 'total_tests': 4, 'tests_passed': 4},  # 100%
            {'task_id': 'math_2', 'category': 'math', 'complexity': 1, 'object_type': 'function', 'total_tests': 2, 'tests_passed': 0},  # 0%
        ]
        absent_solutions = [
            {'task_id': 'math_3', 'category': 'math', 'complexity': 2},
            {'task_id': 'english_1', 'category': 'english', 'complexity': 1},
            {'task_id': 'english_2', 'category': 'english', 'complexity': 1}
        ]
        
        metrics = self.calculator.calculate_all_metrics(evaluation_results, absent_solutions, self.benchmark_data)
        
        # Should have meaningful variance since scores are [100, 0, 0, 0, 0]
        ci = metrics['average_score']['confidence_interval']
        self.assertEqual(ci['mean'], 20.0)  # (100 + 0 + 0 + 0 + 0) / 5 = 20.0
        self.assertGreater(ci['std_dev'], 0)  # Should have non-zero standard deviation
        self.assertGreater(ci['margin_of_error'], 0)  # Should have non-zero margin of error
        self.assertLess(ci['lower_bound'], ci['mean'])  # Lower bound should be less than mean
        self.assertGreater(ci['upper_bound'], ci['mean'])  # Upper bound should be greater than mean


if __name__ == '__main__':
    unittest.main(verbosity=2)