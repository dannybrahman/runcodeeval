#!/usr/bin/env python3
"""
Integration tests for the full evaluation pipeline
"""

import unittest
import sys
import os
import tempfile
import json
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main


class TestIntegration(unittest.TestCase):
    """Integration tests for the full evaluation system"""
    
    def setUp(self):
        """Set up test data files"""
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark_dir = Path(self.temp_dir) / 'benchmark'
        self.solutions_dir = Path(self.temp_dir) / 'solutions'
        self.output_dir = Path(self.temp_dir) / 'output'
        
        # Create directories
        self.benchmark_dir.mkdir()
        self.solutions_dir.mkdir()
        
        # Create test benchmark data
        benchmark_data = [
            {
                "task_id": "test_1",
                "name": "add_numbers", 
                "problem": "Add two numbers",
                "canonical_solution": "def add_numbers(a, b):\n    return a + b",
                "tests": '[{"assertion": "func(2, 3) == 5"}, {"assertion": "func(0, 0) == 0"}]',
                "topic": "math",
                "object": "function",
                "complexity": 1
            },
            {
                "task_id": "test_2",
                "name": "Person",
                "problem": "Create Person class", 
                "canonical_solution": "class Person:\n    def __init__(self, name):\n        self.name = name",
                "tests": '[{"ctx": "p = cls(\\"Alice\\")", "assertion": "p.name == \\"Alice\\""}]',
                "topic": "classes",
                "object": "class", 
                "complexity": 2
            },
            {
                "task_id": "test_3",
                "name": "multiply",
                "problem": "Multiply two numbers",
                "canonical_solution": "def multiply(a, b):\n    return a * b", 
                "tests": '[{"assertion": "func(3, 4) == 12"}]',
                "topic": "math",
                "object": "function",
                "complexity": 1
            }
        ]
        
        with open(self.benchmark_dir / 'problems.jsonl', 'w') as f:
            for problem in benchmark_data:
                f.write(json.dumps(problem) + '\n')
        
        # Create test solution data - student attempts 2 out of 3, gets both correct
        solution_data = [
            {
                "task_id": "test_1_student",
                "model": "student",
                "candidate_solution": "def add_numbers(a, b):\n    return a + b"
            },
            {
                "task_id": "test_2_student", 
                "model": "student",
                "candidate_solution": "class Person:\n    def __init__(self, name):\n        self.name = name"
            }
            # Note: test_3 is absent (not attempted)
        ]
        
        with open(self.solutions_dir / 'solutions.jsonl', 'w') as f:
            for solution in solution_data:
                f.write(json.dumps(solution) + '\n')
    
    def tearDown(self):
        """Clean up temp files"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_absent_vs_nocompletion_distinction(self):
        """Test the distinction between absent solutions and NoCompletionError"""
        # Create benchmark with 3 problems
        benchmark_data = main.load_benchmark_data(self.benchmark_dir)
        
        # Create solution data with:
        # - test_1: valid solution 
        # - test_2: empty solution (NoCompletionError)
        # - test_3: absent (not in solution file at all)
        solution_data = [
            {
                "task_id": "test_1_student",
                "model": "student", 
                "candidate_solution": "def add_numbers(a, b):\n    return a + b"
            },
            {
                "task_id": "test_2_student",
                "model": "student",
                "candidate_solution": ""  # Empty solution = NoCompletionError
            }
            # test_3 is absent (not in solution file)
        ]
        
        with open(self.solutions_dir / 'solutions.jsonl', 'w') as f:
            for solution in solution_data:
                f.write(json.dumps(solution) + '\n')
        
        # Load and evaluate
        solution_data, model_name = main.load_solution_data(self.solutions_dir)
        result = main.evaluate_solutions(
            benchmark_data, solution_data, model_name, 
            self.output_dir, verbose=False
        )
        
        # Verify metrics
        self.assertEqual(result['total_evaluated'], 2)  # test_1 + test_2 (empty)
        self.assertEqual(result['total_absent'], 1)     # test_3
        
        # Check error analysis - should only count NoCompletionError, not absent
        error_analysis = result['error_analysis']
        if 'NoCompletionError' in error_analysis['error_counts']:
            self.assertEqual(error_analysis['error_counts']['NoCompletionError'], 1)  # Only test_2
        
        # Total errors should be 1 (test_2 empty), not 2 (would include absent test_3)
        self.assertEqual(error_analysis['total_errors'], 1)

    def test_end_to_end_evaluation(self):
        """Test complete evaluation pipeline"""
        # Load benchmark and solution data
        benchmark_data = main.load_benchmark_data(self.benchmark_dir)
        solution_data, model_name = main.load_solution_data(self.solutions_dir)
        
        # Run evaluation
        result = main.evaluate_solutions(
            benchmark_data, 
            solution_data, 
            model_name,
            self.output_dir,
            verbose=False
        )
        
        # Verify results
        self.assertEqual(result['model'], 'student')
        self.assertEqual(result['total_evaluated'], 2)
        self.assertEqual(result['total_absent'], 1)
        
        # Verify metrics are correctly calculated
        metrics = result['metrics']
        
        # Expected: (100 + 100 + 0) / 3 = 66.67%
        avg_score = metrics['average_score']
        if isinstance(avg_score, dict):
            self.assertAlmostEqual(avg_score['score'], 66.67, places=2)
            # Check that confidence interval exists
            self.assertIn('confidence_interval', avg_score)
            self.assertIn('margin_of_error', avg_score['confidence_interval'])
        else:
            self.assertAlmostEqual(avg_score, 66.67, places=2)
        
        # Math category: (100 + 0) / 2 = 50%
        math_score = metrics['score_by_category']['math']
        if isinstance(math_score, dict):
            self.assertEqual(math_score['score'], 50.0)
        else:
            self.assertEqual(math_score, 50.0)
        
        # Classes category: (100) / 1 = 100%
        classes_score = metrics['score_by_category']['classes']
        if isinstance(classes_score, dict):
            self.assertEqual(classes_score['score'], 100.0)
        else:
            self.assertEqual(classes_score, 100.0)
        
        # Complexity level 1: (100 + 0) / 2 = 50%
        complexity1_score = metrics['score_by_complexity'][1]
        if isinstance(complexity1_score, dict):
            self.assertEqual(complexity1_score['score'], 50.0)
        else:
            self.assertEqual(complexity1_score, 50.0)
        
        # Complexity level 2: (100) / 1 = 100%
        complexity2_score = metrics['score_by_complexity'][2]
        if isinstance(complexity2_score, dict):
            self.assertEqual(complexity2_score['score'], 100.0)
        else:
            self.assertEqual(complexity2_score, 100.0)
        
        # Function type: (100 + 0) / 2 = 50%
        function_score = metrics['score_by_object']['function']
        if isinstance(function_score, dict):
            self.assertEqual(function_score['score'], 50.0)
        else:
            self.assertEqual(function_score, 50.0)
        
        # Class type: (100) / 1 = 100%
        class_score = metrics['score_by_object']['class']
        if isinstance(class_score, dict):
            self.assertEqual(class_score['score'], 100.0)
        else:
            self.assertEqual(class_score, 100.0)
        
        # Verify output files were created in the output directory
        # Note: Integration test calls evaluate_solutions directly, not through CLI,
        # so files are created directly in the output directory (no session structure)
        self.assertTrue((self.output_dir / 'test_results_score.json').exists())
        self.assertTrue((self.output_dir / 'test_results_errors.json').exists())
        self.assertTrue((self.output_dir / 'evaluation_summary.md').exists())
        self.assertTrue((self.output_dir / 'test_results.jsonl').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)