"""
Core evaluation engine that executes test cases against solutions
Based on the dataset project's test validation approach
"""

import json
import ast
import traceback
import signal
from typing import Dict, Any, List, Tuple
from contextlib import contextmanager
import logging


class TimeoutException(Exception):
    """Custom exception for timeout handling"""
    pass


@contextmanager
def timeout(seconds: int):
    """Context manager for timeout handling"""
    def timeout_handler(signum, frame):
        raise TimeoutException("Execution timed out")
    
    # Set up the timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class CoreEvaluator:
    """Core evaluation engine for testing solutions against benchmark problems"""
    
    def __init__(self, timeout_seconds: int = 60):
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(__name__)
    
    def evaluate_single_solution(self, 
                               benchmark_problem: Dict[str, Any],
                               candidate_solution: str,
                               task_id: str) -> Dict[str, Any]:
        """
        Evaluate a single solution against its benchmark problem
        
        Returns evaluation result with test outcomes and error details
        """
        self.logger.debug(f"Starting evaluation for task: {task_id}")
        
        # Initialize result
        result = {
            'task_id': task_id,
            'category': benchmark_problem.get('topic', 'unknown'),
            'complexity': benchmark_problem.get('complexity', 0),
            'object_type': benchmark_problem.get('object', 'unknown'),
            'total_tests': 0,
            'tests_passed': 0,
            'all_tests_passed': False,
            'error_type': None,
            'error_message': None,
            'test_details': []
        }
        
        # Check for empty solution
        if not candidate_solution or not candidate_solution.strip():
            result['error_type'] = 'NoCompletionError'
            result['error_message'] = 'Empty solution provided'
            return result
        
        # Parse tests
        try:
            tests_data = benchmark_problem.get('tests', '[]')
            if isinstance(tests_data, str):
                tests = json.loads(tests_data)
            elif isinstance(tests_data, list):
                tests = tests_data
            else:
                raise ValueError(f"Invalid tests format: {type(tests_data)}")
            
            result['total_tests'] = len(tests)
        except Exception as e:
            result['error_type'] = 'TestParseError'
            result['error_message'] = f"Failed to parse tests: {str(e)}"
            return result
        
        # Execute solution with timeout
        try:
            with timeout(self.timeout_seconds):
                # Create execution namespace
                namespace = {'__builtins__': __builtins__}
                
                # Execute the candidate solution
                try:
                    # Temporarily disable logging to avoid socket handler errors from tested code
                    logging.disable(logging.CRITICAL)
                    exec(candidate_solution, namespace)
                    logging.disable(logging.NOTSET)
                except SyntaxError as e:
                    result['error_type'] = 'SyntaxError'
                    result['error_message'] = f"Syntax error at line {e.lineno}: {str(e)}"
                    return result
                except Exception as e:
                    result['error_type'] = 'ExecutionError'
                    result['error_message'] = f"Failed to execute solution: {str(e)}"
                    return result
                
                # Make solution source available for inspection tests
                namespace['candidate_solution'] = candidate_solution
                
                # Set up function/class references
                name = benchmark_problem.get('name', 'unknown')
                if benchmark_problem.get('object') == 'function':
                    func_name = name
                    if func_name not in namespace:
                        # Try to find any function
                        func_candidates = [k for k, v in namespace.items() 
                                         if callable(v) and not k.startswith('_')]
                        if func_candidates:
                            func_name = func_candidates[0]
                        else:
                            result['error_type'] = 'NameError'
                            result['error_message'] = f"Function '{name}' not found in solution"
                            return result
                    namespace['func'] = namespace[func_name]
                    
                elif benchmark_problem.get('object') == 'class':
                    class_name = name
                    if class_name not in namespace:
                        # Try to find any class
                        class_candidates = [k for k, v in namespace.items() 
                                          if isinstance(v, type) and not k.startswith('_')]
                        if class_candidates:
                            class_name = class_candidates[0]
                        else:
                            result['error_type'] = 'NameError'
                            result['error_message'] = f"Class '{name}' not found in solution"
                            return result
                    namespace['cls'] = namespace[class_name]
                
                # Run each test
                for i, test in enumerate(tests):
                    test_result = self._run_single_test(test, namespace, i)
                    result['test_details'].append(test_result)
                    
                    if test_result['passed']:
                        result['tests_passed'] += 1
        
        except TimeoutException:
            result['error_type'] = 'TimeoutError'
            result['error_message'] = f'Execution exceeded {self.timeout_seconds} seconds'
            return result
        except Exception as e:
            result['error_type'] = 'Error'
            result['error_message'] = f"Unexpected error: {str(e)}"
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return result
        
        # Calculate final result
        result['all_tests_passed'] = (result['tests_passed'] == result['total_tests'] 
                                     and result['total_tests'] > 0)
        
        # Determine error type if tests failed
        if not result['all_tests_passed'] and result['error_type'] is None:
            # Check test details for common error patterns
            for detail in result['test_details']:
                if not detail['passed'] and detail.get('error_type'):
                    result['error_type'] = detail['error_type']
                    break
            
            if result['error_type'] is None:
                result['error_type'] = 'TestFailure'
        
        return result
    
    def _run_single_test(self, test: Dict[str, Any], namespace: Dict[str, Any], test_index: int) -> Dict[str, Any]:
        """Run a single test case"""
        self.logger.debug(f"Running test {test_index}: {test.get('assertion', test.get('inspection', 'unknown'))[:100]}")
        
        test_result = {
            'test_index': test_index,
            'passed': False,
            'error_type': None,
            'error_message': None
        }
        
        try:
            # Create test-specific namespace
            test_namespace = namespace.copy()
            
            # Execute context setup if present
            if test.get('ctx'):
                logging.disable(logging.CRITICAL)
                exec(test['ctx'], test_namespace)
                logging.disable(logging.NOTSET)
            
            # Handle inspection tests
            if 'inspection' in test:
                passed, error_msg = self._run_inspection_test(
                    test['inspection'], 
                    namespace.get('candidate_solution', '')
                )
                test_result['passed'] = passed
                if not passed:
                    test_result['error_message'] = error_msg
                    test_result['error_type'] = 'InspectionFailure'
            
            # Handle assertion tests
            if 'assertion' in test:
                assertion = test['assertion']
                try:
                    logging.disable(logging.CRITICAL)
                    result = eval(assertion, test_namespace)
                    logging.disable(logging.NOTSET)
                    if not result:
                        test_result['error_message'] = f"Assertion failed: {assertion}"
                        test_result['error_type'] = 'AssertionError'
                    else:
                        test_result['passed'] = True
                except NameError as e:
                    test_result['error_type'] = 'NameError'
                    test_result['error_message'] = str(e)
                except TypeError as e:
                    test_result['error_type'] = 'TypeError'
                    test_result['error_message'] = str(e)
                except Exception as e:
                    test_result['error_type'] = type(e).__name__
                    test_result['error_message'] = str(e)
        
        except Exception as e:
            test_result['error_type'] = 'Error'
            test_result['error_message'] = f"Test execution error: {str(e)}"
        
        return test_result
    
    def _run_inspection_test(self, keywords: Any, source_code: str) -> Tuple[bool, str]:
        """Run AST-based inspection test"""
        if not isinstance(keywords, list):
            keywords = [keywords]
        
        try:
            tree = ast.parse(source_code)
            
            for keyword in keywords:
                if not self._find_keyword_in_ast(tree, keyword):
                    return False, f"Required keyword '{keyword}' not found in implementation"
            
            return True, None
            
        except SyntaxError as e:
            return False, f"Syntax error in solution: {str(e)}"
        except Exception as e:
            return False, f"Inspection error: {str(e)}"
    
    def _find_keyword_in_ast(self, tree: ast.AST, keyword: str) -> bool:
        """Find a keyword in the AST"""
        # Handle module.function patterns
        if '.' in keyword:
            module_name, func_name = keyword.rsplit('.', 1)
            
            for node in ast.walk(tree):
                # Check imports
                if isinstance(node, ast.ImportFrom) and node.module == module_name:
                    for alias in node.names:
                        if alias.name == func_name:
                            return True
                
                # Check direct usage
                elif isinstance(node, ast.Call) and hasattr(node.func, 'attr'):
                    if node.func.attr == func_name:
                        return True
                
                # Check function calls after import
                elif isinstance(node, ast.Call) and hasattr(node.func, 'id'):
                    if node.func.id == func_name:
                        # Verify it was imported
                        for import_node in ast.walk(tree):
                            if isinstance(import_node, ast.ImportFrom) and import_node.module == module_name:
                                for alias in import_node.names:
                                    if alias.name == func_name:
                                        return True
        
        else:
            # Handle special keywords
            if keyword == "yield from":
                for node in ast.walk(tree):
                    if isinstance(node, ast.YieldFrom):
                        return True
            else:
                # Simple keyword check
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        if hasattr(node.func, 'id') and node.func.id == keyword:
                            return True
                        elif hasattr(node.func, 'attr') and node.func.attr == keyword:
                            return True
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        for alias in node.names:
                            if alias.name == keyword:
                                return True
        
        return False