"""Storage system for collected LLM solutions"""

import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import pandas as pd

class SolutionStore:
    """Handles storage and retrieval of collected LLM solutions"""
    
    def __init__(self, output_dir: str, session_id: str = None, is_resume: bool = False):
        """
        Initialize the solution store
        
        Args:
            output_dir: Directory to store collected solutions
            session_id: Session identifier for organizing model-specific folders
            is_resume: Whether this is a resume session (affects how solutions are saved)
        """
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(exist_ok=True, parents=True)
        self.session_id = session_id or f"session_{int(time.time())}"
        self.is_resume = is_resume
        
        # Set session directory path (but don't create it here)
        self.output_dir = self.base_output_dir / self.session_id
        
        # Solutions grouped by model and category
        self.solutions_by_model_category: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        self.metadata = {
            "collection_start": None,
            "collection_end": None,
            "total_problems": 0,
            "models_evaluated": [],
            "categories_included": [],
            "session_id": self.session_id,
            "session_parameters": {
                "models": [],
                "categories": None,
                "quick_test": False,
                "dataset_path": None,
                "batch_size": None,
                "max_concurrent_per_model": None,
                "output_dir": None
            }
        }
    
    def set_session_parameters(self, models: List[str], categories: Optional[List[str]] = None, 
                              quick_test: bool = False, dataset_path: str = None,
                              batch_size: int = None, max_concurrent_per_model: int = None,
                              output_dir: str = None):
        """
        Store the original session parameters for resume functionality
        
        Args:
            models: List of model names used in the session
            categories: List of categories processed (None for all)
            quick_test: Whether this was a quick test session
            dataset_path: Path to the dataset used
            batch_size: Batch size used
            max_concurrent_per_model: Concurrency limit used
            output_dir: Output directory where session is stored
        """
        self.metadata["session_parameters"].update({
            "models": models,
            "categories": categories,
            "quick_test": quick_test,
            "dataset_path": dataset_path,
            "batch_size": batch_size,
            "max_concurrent_per_model": max_concurrent_per_model,
            "output_dir": output_dir
        })
        
        # Also update the legacy fields for compatibility
        self.metadata["models_evaluated"] = models
        self.metadata["categories_included"] = categories or []
        
        # Set collection start time
        if self.metadata["collection_start"] is None:
            self.metadata["collection_start"] = time.time()
    
    def set_total_problems(self, total_problems: int):
        """Set the total number of problems being processed"""
        self.metadata["total_problems"] = total_problems

    def add_solutions(self, solutions: List[Dict]):
        """Add a batch of solutions to the store"""
        for solution in solutions:
            model_name = solution.get("model", "unknown")
            category = solution.get("category", "unknown")
            self.solutions_by_model_category[model_name][category].append(solution)
    
    def add_single_solution(self, task_id: str, model_name: str, solution_data: Dict):
        """
        Add a single solution result
        
        Args:
            task_id: Problem identifier
            model_name: Model that generated the solution
            solution_data: Solution data including code, validation results, etc.
        """
        solution_record = {
            "task_id": task_id,
            "model": model_name,
            "timestamp": time.time(),
            **solution_data
        }
        category = solution_data.get("category", "unknown")
        self.solutions_by_model_category[model_name][category].append(solution_record)
    
    def _create_model_directory(self, model_name: str) -> Path:
        """Create directory for model-specific solutions"""
        model_dir = self.output_dir / model_name
        model_dir.mkdir(exist_ok=True, parents=True)
        return model_dir
    
    def save_solutions_jsonl(self) -> Dict[str, Dict[str, Path]]:
        """
        Save solutions in JSONL format, one file per category per model
        
        Returns:
            Dictionary mapping model names to dictionaries of category -> file path
        """
        output_paths = {}
        
        for model_name, categories in self.solutions_by_model_category.items():
            if not categories:
                continue
                
            model_dir = self._create_model_directory(model_name)
            model_paths = {}
            
            for category, solutions in categories.items():
                if not solutions:
                    continue
                    
                filename = f"{category}.jsonl"
                output_path = model_dir / filename
                
                # For resume sessions, load existing solutions first
                if self.is_resume and output_path.exists():
                    existing_solutions = self.load_solutions_from_jsonl(model_name, category)
                    
                    # Create a set of existing task_ids to avoid duplicates
                    existing_task_ids = {sol.get('task_id') for sol in existing_solutions}
                    
                    # Merge: existing solutions + new solutions (avoiding duplicates)
                    merged_solutions = existing_solutions.copy()
                    for new_sol in solutions:
                        if new_sol.get('task_id') not in existing_task_ids:
                            merged_solutions.append(new_sol)
                    
                    solutions = merged_solutions
                
                # Write all solutions (either new or merged)
                with open(output_path, 'w') as f:
                    for solution in solutions:
                        f.write(json.dumps(solution) + "\n")
                
                model_paths[category] = output_path
            
            output_paths[model_name] = model_paths
        
        return output_paths
    
    def save_solutions_csv(self) -> Dict[str, Dict[str, Path]]:
        """
        Save solutions in CSV format, one file per category per model
        
        Returns:
            Dictionary mapping model names to dictionaries of category -> file path
        """
        output_paths = {}
        
        for model_name, categories in self.solutions_by_model_category.items():
            if not categories:
                continue
                
            model_dir = self._create_model_directory(model_name)
            model_paths = {}
            
            for category, solutions in categories.items():
                if not solutions:
                    continue
                    
                filename = f"{category}.csv"
                output_path = model_dir / filename
                
                # For resume sessions, load and merge existing solutions first
                if self.is_resume and output_path.exists():
                    # Load existing CSV
                    existing_df = pd.read_csv(output_path)
                    existing_task_ids = set(existing_df['task_id'].values) if 'task_id' in existing_df.columns else set()
                    
                    # Convert new solutions to DataFrame
                    new_df = pd.DataFrame(solutions)
                    
                    # Filter out duplicates based on task_id
                    if 'task_id' in new_df.columns:
                        new_df = new_df[~new_df['task_id'].isin(existing_task_ids)]
                    
                    # Merge dataframes
                    df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    # Convert to pandas DataFrame for easier CSV handling
                    df = pd.DataFrame(solutions)
                
                df.to_csv(output_path, index=False)
                
                model_paths[category] = output_path
            
            output_paths[model_name] = model_paths
        
        return output_paths
    
    def save_model_summaries(self, filename: str = "summary.json") -> Path:
        """
        Save combined analysis summary with statistics for all models at session level
        
        Args:
            filename: Output filename
            
        Returns:
            Path to the session-level summary file
        """
        # Update collection end time in metadata
        self.metadata["collection_end"] = time.time()
        
        # Generate summaries for each model
        model_summaries = {}
        
        for model_name, categories in self.solutions_by_model_category.items():
            if not categories:
                continue
            
            # Flatten all solutions for this model
            all_solutions = []
            for category_solutions in categories.values():
                all_solutions.extend(category_solutions)
            
            model_summaries[model_name] = self._generate_model_summary(model_name, all_solutions, categories)
        
        # Create session-level summary
        session_summary = {
            "session_id": self.session_id,
            "collection_metadata": self.metadata,
            "model_summaries": model_summaries,
            "collection_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save at session level (not in model directory)
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(session_summary, f, indent=2)
        
        return output_path
    
    def _group_by_task_id(self, solutions: List[Dict] = None) -> Dict[str, List[Dict]]:
        """Group solutions by task_id"""
        if solutions is None:
            # Group all solutions across all models and categories
            all_solutions = []
            for model_categories in self.solutions_by_model_category.values():
                for category_solutions in model_categories.values():
                    all_solutions.extend(category_solutions)
            solutions = all_solutions
            
        grouped = defaultdict(list)
        for solution in solutions:
            task_id = solution.get("task_id", "unknown")
            grouped[task_id].append(solution)
        return dict(grouped)
    
    def _generate_model_summary(self, model_name: str, solutions: List[Dict], categories: Dict[str, List[Dict]] = None) -> Dict:
        """Generate summary statistics for a specific model"""
        if not solutions:
            return {"error": "No solutions collected for this model"}
        
        grouped = self._group_by_task_id(solutions)
        
        # Basic stats
        total_problems = len(grouped)
        total_solutions = len(solutions)
        
        # Performance stats
        stats = {
            "total_attempts": 0,
            "successful_extractions": 0,
            "valid_syntax": 0,
            "has_expected_name": 0,
            "avg_execution_time": 0.0,
            "errors": 0
        }
        
        for solution in solutions:
            stats["total_attempts"] += 1
            
            if solution.get("extracted_solution"):
                stats["successful_extractions"] += 1
            
            if solution.get("is_valid_syntax"):
                stats["valid_syntax"] += 1
            
            if solution.get("has_expected_name"):
                stats["has_expected_name"] += 1
            
            if solution.get("execution_time"):
                stats["avg_execution_time"] += solution["execution_time"]
            
            if solution.get("error"):
                stats["errors"] += 1
        
        # Calculate averages and rates
        if stats["total_attempts"] > 0:
            stats["avg_execution_time"] /= stats["total_attempts"]
            stats["success_rate"] = stats["successful_extractions"] / stats["total_attempts"] * 100
            stats["syntax_validity_rate"] = stats["valid_syntax"] / stats["total_attempts"] * 100
            stats["name_accuracy_rate"] = stats["has_expected_name"] / stats["total_attempts"] * 100
            stats["error_rate"] = stats["errors"] / stats["total_attempts"] * 100
        
        # Category distribution
        category_counts = defaultdict(int)
        complexities = defaultdict(int)
        
        for solution in solutions:
            if "category" in solution:
                category_counts[solution["category"]] += 1
            if "complexity" in solution:
                complexities[solution["complexity"]] += 1
        
        # Per-category performance if categories provided
        category_performance = {}
        if categories:
            for category_name, category_solutions in categories.items():
                if category_solutions:
                    category_stats = self._calculate_category_stats(category_solutions)
                    category_performance[category_name] = category_stats
        
        summary = {
            "model_name": model_name,
            "overview": {
                "total_problems": total_problems,
                "total_solutions": total_solutions,
                "categories_processed": list(categories.keys()) if categories else []
            },
            "overall_performance": stats,
            "category_performance": category_performance,
            "problem_distribution": {
                "by_category": dict(category_counts),
                "by_complexity": dict(complexities)
            }
        }
        
        return summary
    
    def _calculate_category_stats(self, solutions: List[Dict]) -> Dict:
        """Calculate performance statistics for a specific category"""
        stats = {
            "total_attempts": len(solutions),
            "successful_extractions": 0,
            "valid_syntax": 0,
            "has_expected_name": 0,
            "avg_execution_time": 0.0,
            "errors": 0
        }
        
        for solution in solutions:
            if solution.get("extracted_solution"):
                stats["successful_extractions"] += 1
            if solution.get("is_valid_syntax"):
                stats["valid_syntax"] += 1
            if solution.get("has_expected_name"):
                stats["has_expected_name"] += 1
            if solution.get("execution_time"):
                stats["avg_execution_time"] += solution["execution_time"]
            if solution.get("error"):
                stats["errors"] += 1
        
        # Calculate rates
        if stats["total_attempts"] > 0:
            stats["avg_execution_time"] /= stats["total_attempts"]
            stats["success_rate"] = stats["successful_extractions"] / stats["total_attempts"] * 100
            stats["syntax_validity_rate"] = stats["valid_syntax"] / stats["total_attempts"] * 100
            stats["name_accuracy_rate"] = stats["has_expected_name"] / stats["total_attempts"] * 100
            stats["error_rate"] = stats["errors"] / stats["total_attempts"] * 100
        
        return stats
    
    def load_solutions_from_jsonl(self, model_name: str, category: str) -> List[Dict]:
        """Load solutions from a specific model's category JSONL file"""
        model_dir = self.output_dir / model_name
        filepath = model_dir / f"{category}.jsonl"
        
        if not filepath.exists():
            return []
        
        solutions = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    solutions.append(json.loads(line))
        return solutions
    
    def get_category_files(self, model_name: str) -> List[str]:
        """Get list of category files for a specific model"""
        model_dir = self.output_dir / model_name
        if not model_dir.exists():
            return []
        
        category_files = []
        for jsonl_file in model_dir.glob("*.jsonl"):
            category_files.append(jsonl_file.stem)
        return sorted(category_files)
    
    def save_cross_model_comparison(self, filename: str = "cross_model_comparison.json") -> Path:
        """Save cross-model comparison data to the session directory"""
        output_path = self.output_dir / filename
        
        grouped = self._group_by_task_id()
        comparison_data = []
        
        for task_id, task_solutions in grouped.items():
            if not task_solutions:
                continue
            
            # Get problem metadata
            problem_data = {
                "task_id": task_id,
                "problem": task_solutions[0].get("problem", ""),
                "category": task_solutions[0].get("category", ""),
                "complexity": task_solutions[0].get("complexity", 0)
            }
            
            # Add model results
            for solution in task_solutions:
                model = solution["model"]
                problem_data[f"{model}_valid"] = solution.get("is_valid_syntax", False)
                problem_data[f"{model}_has_name"] = solution.get("has_expected_name", False)
                problem_data[f"{model}_time"] = solution.get("execution_time", 0.0)
                problem_data[f"{model}_error"] = bool(solution.get("error"))
            
            comparison_data.append(problem_data)
        
        # Generate overall summary
        overall_summary = {
            "session_id": self.session_id,
            "models_evaluated": list(self.solutions_by_model_category.keys()),
            "categories_processed": self._get_all_categories(),
            "total_problems": len(comparison_data),
            "collection_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_summaries": {}
        }
        
        # Add per-model summaries
        for model_name, categories in self.solutions_by_model_category.items():
            all_solutions = []
            for category_solutions in categories.values():
                all_solutions.extend(category_solutions)
            overall_summary["model_summaries"][model_name] = self._generate_model_summary(model_name, all_solutions, categories)
        
        final_data = {
            "summary": overall_summary,
            "problems": comparison_data
        }
        
        with open(output_path, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        return output_path
    
    def get_category_comparison_data(self, category: str) -> Dict:
        """Get comparison data for a specific category across all models"""
        comparison_data = []
        
        # Get all solutions for this category across all models
        category_solutions = []
        for model_name, categories in self.solutions_by_model_category.items():
            if category in categories:
                category_solutions.extend(categories[category])
        
        # Group by task_id for comparison
        grouped = self._group_by_task_id(category_solutions)
        
        for task_id, task_solutions in grouped.items():
            if not task_solutions:
                continue
            
            # Get problem metadata
            problem_data = {
                "task_id": task_id,
                "problem": task_solutions[0].get("problem", ""),
                "category": category,
                "complexity": task_solutions[0].get("complexity", 0)
            }
            
            # Add model results
            for solution in task_solutions:
                model = solution["model"]
                problem_data[f"{model}_valid"] = solution.get("is_valid_syntax", False)
                problem_data[f"{model}_has_name"] = solution.get("has_expected_name", False)
                problem_data[f"{model}_time"] = solution.get("execution_time", 0.0)
                problem_data[f"{model}_error"] = bool(solution.get("error"))
            
            comparison_data.append(problem_data)
        
        return {
            "category": category,
            "problems": comparison_data,
            "models_with_data": [model for model, categories in self.solutions_by_model_category.items() if category in categories]
        }
    
    def _get_all_categories(self) -> List[str]:
        """Get all unique categories across all models"""
        categories = set()
        for model_categories in self.solutions_by_model_category.values():
            categories.update(model_categories.keys())
        return sorted(list(categories))
    
    @staticmethod
    def load_session_parameters(session_id: str, output_dir: str) -> Optional[Dict]:
        """
        Load session parameters from session-level summary.json file
        
        Args:
            session_id: Session ID to load parameters for
            output_dir: Base output directory containing sessions
            
        Returns:
            Dictionary containing session parameters, or None if not found
        """
        session_dir = Path(output_dir) / session_id
        if not session_dir.exists():
            return None
        
        # Look for session-level summary.json file
        summary_file = session_dir / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                
                # Extract session parameters from metadata
                metadata = summary_data.get("collection_metadata", {})
                session_params = metadata.get("session_parameters", {})
                
                return session_params if session_params else None
                
            except (json.JSONDecodeError, KeyError):
                return None
        
        return None
