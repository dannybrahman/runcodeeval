"""
Report generator for creating evaluation reports
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging


class ReportGenerator:
    """Generate evaluation reports in various formats"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def _format_score_with_ci(self, label: str, score_data) -> str:
        """Format a score with confidence interval for display"""
        if isinstance(score_data, dict) and 'confidence_interval' in score_data:
            score = score_data['score']
            ci = score_data['confidence_interval']
            return f"- **{label}**: {score:.2f}% ± {ci['margin_of_error']:.2f}% (95% CI: {ci['lower_bound']:.2f}%-{ci['upper_bound']:.2f}%)"
        else:
            # Backward compatibility - assume it's just a number
            return f"- **{label}**: {score_data:.2f}%"
    
    def generate_evaluation_report(self,
                                 model_name: str,
                                 evaluation_results: List[Dict[str, Any]],
                                 metrics: Dict[str, Any],
                                 error_analysis: Dict[str, Any],
                                 output_dir: Path):
        """Generate comprehensive evaluation report for a model"""
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create score report (similar to test_results_score.json)
        score_report = {
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'total': metrics['average_score'],
            'by_topic': metrics['score_by_category'],
            'by_complexity': {
                f'level_{k}': v for k, v in metrics['score_by_complexity'].items()
            },
            'by_problem_type': metrics['score_by_object'],
            'summary': {
                'total_problems': metrics['total_problems'],
                'solutions_provided': metrics['solutions_provided'],
                'solutions_absent': metrics['solutions_absent'],
                'overall_stats': metrics['overall_stats']
            }
        }
        
        # Save score report
        score_file = output_dir / 'test_results_score.json'
        with open(score_file, 'w') as f:
            json.dump(score_report, f, indent=2)
        
        # Create error report (similar to test_results_errors.json)
        error_report = {
            'model': model_name,
            'timestamp': datetime.now().isoformat(),
            'total_errors': error_analysis['total_errors'],
            'error_rate': error_analysis['error_rate'],
            'error_breakdown': error_analysis['error_percentages'],
            'error_counts': error_analysis['error_counts'],
            'by_topic': error_analysis['error_by_category'],
            'by_complexity': {
                f'level_{k}': v for k, v in error_analysis['error_by_complexity'].items()
            },
            'by_problem_type': error_analysis['error_by_object'],
            'category_error_rates': error_analysis['category_error_rates'],
            'complexity_error_rates': {
                f'level_{k}': v for k, v in error_analysis['complexity_error_rates'].items()
            },
            'error_examples': error_analysis['error_examples']
        }
        
        # Save error report
        error_file = output_dir / 'test_results_errors.json'
        with open(error_file, 'w') as f:
            json.dump(error_report, f, indent=2)
        
        # Generate human-readable summary
        self._generate_summary_report(model_name, metrics, error_analysis, output_dir)
        
        self.logger.info(f"Reports generated in: {output_dir}")
    
    def _generate_summary_report(self, 
                               model_name: str,
                               metrics: Dict[str, Any],
                               error_analysis: Dict[str, Any],
                               output_dir: Path):
        """Generate human-readable summary report"""
        
        summary_lines = [
            f"# Evaluation Report for {model_name}",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## Overall Performance",
            self._format_score_with_ci("Total Score", metrics['average_score']),
            f"- **Problems in Benchmark**: {metrics['total_problems']}",
            f"- **Solutions Provided**: {metrics['solutions_provided']}",
            f"- **Solutions Absent**: {metrics['solutions_absent']}",
            f"\n## Performance Breakdown",
            f"\n### By Category",
        ]
        
        # Add category scores
        for category, score_data in sorted(metrics['score_by_category'].items()):
            if isinstance(score_data, dict):
                score = score_data['score']
                ci = score_data['confidence_interval']
                summary_lines.append(f"- {category}: {score:.2f}% ± {ci['margin_of_error']:.2f}%")
            else:
                summary_lines.append(f"- {category}: {score_data:.2f}%")
        
        summary_lines.extend([
            f"\n### By Complexity",
        ])
        
        # Add complexity scores
        for level in [1, 2, 3]:
            level_name = ["Easy", "Medium", "Hard"][level-1]
            score_data = metrics['score_by_complexity'][level]
            if isinstance(score_data, dict):
                score = score_data['score']
                ci = score_data['confidence_interval']
                summary_lines.append(f"- Level {level} ({level_name}): {score:.2f}% ± {ci['margin_of_error']:.2f}%")
            else:
                summary_lines.append(f"- Level {level} ({level_name}): {score_data:.2f}%")
        
        summary_lines.append(f"\n### By Problem Type")
        
        # Add object type scores
        for obj_type, score_data in sorted(metrics['score_by_object'].items()):
            if isinstance(score_data, dict):
                score = score_data['score']
                ci = score_data['confidence_interval']
                summary_lines.append(f"- {obj_type}: {score:.2f}% ± {ci['margin_of_error']:.2f}%")
            else:
                summary_lines.append(f"- {obj_type}: {score_data:.2f}%")
        
        # Add error analysis
        summary_lines.extend([
            f"\n## Error Analysis",
            f"- **Total Errors**: {error_analysis['total_errors']}",
            f"- **Error Rate**: {error_analysis['error_rate']:.2f}%",
            f"\n### Error Types"
        ])
        
        # Add error breakdown
        for error_type, percentage in sorted(error_analysis['error_percentages'].items(), 
                                           key=lambda x: x[1], reverse=True):
            count = error_analysis['error_counts'][error_type]
            summary_lines.append(f"- {error_type}: {count} ({percentage:.2f}%)")
        
        # Add solution statistics
        stats = metrics['overall_stats']
        summary_lines.extend([
            f"\n## Solution Statistics",
            f"- **Fully Correct**: {stats['fully_correct']} "
            f"({stats['fully_correct']/stats['total_evaluated']*100:.1f}%)" if stats['total_evaluated'] > 0 else "",
            f"- **Partially Correct**: {stats['partially_correct']} "
            f"({stats['partially_correct']/stats['total_evaluated']*100:.1f}%)" if stats['total_evaluated'] > 0 else "",
            f"- **Incorrect**: {stats['incorrect']} "
            f"({stats['incorrect']/stats['total_evaluated']*100:.1f}%)" if stats['total_evaluated'] > 0 else "",
            f"- **Syntax Errors**: {stats['syntax_errors']}",
            f"- **Runtime Errors**: {stats['runtime_errors']}",
            f"- **Timeout Errors**: {stats['timeout_errors']}"
        ])
        
        # Save summary
        summary_file = output_dir / 'evaluation_summary.md'
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))
    
    def generate_comparative_report(self, all_results: List[Dict[str, Any]], output_dir: Path):
        """Generate comparative report across multiple models"""
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'models_evaluated': [r['model'] for r in all_results],
            'overall_scores': {},
            'category_comparison': defaultdict(dict),
            'complexity_comparison': defaultdict(dict),
            'object_type_comparison': defaultdict(dict),
            'error_rate_comparison': {}
        }
        
        # Collect scores for each model
        for result in all_results:
            model = result['model']
            metrics = result['metrics']
            error_analysis = result['error_analysis']
            
            # Overall score
            comparison['overall_scores'][model] = metrics['average_score']
            
            # Category scores
            for category, score in metrics['score_by_category'].items():
                comparison['category_comparison'][category][model] = score
            
            # Complexity scores
            for complexity, score in metrics['score_by_complexity'].items():
                comparison['complexity_comparison'][complexity][model] = score
            
            # Object type scores
            for obj_type, score in metrics['score_by_object'].items():
                comparison['object_type_comparison'][obj_type][model] = score
            
            # Error rate
            comparison['error_rate_comparison'][model] = error_analysis['error_rate']
        
        # Convert defaultdicts to regular dicts
        comparison['category_comparison'] = dict(comparison['category_comparison'])
        comparison['complexity_comparison'] = dict(comparison['complexity_comparison'])
        comparison['object_type_comparison'] = dict(comparison['object_type_comparison'])
        
        # Save comparison
        comparison_file = output_dir / 'model_comparison.json'
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Generate comparison summary
        self._generate_comparison_summary(comparison, output_dir)
    
    def _generate_comparison_summary(self, comparison: Dict[str, Any], output_dir: Path):
        """Generate human-readable comparison summary"""
        
        summary_lines = [
            "# Model Comparison Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## Models Evaluated",
        ]
        
        # List models
        for model in comparison['models_evaluated']:
            summary_lines.append(f"- {model}")
        
        # Overall scores
        summary_lines.append("\n## Overall Scores")
        sorted_models = sorted(comparison['overall_scores'].items(), 
                             key=lambda x: x[1]['score'] if isinstance(x[1], dict) else x[1], 
                             reverse=True)
        
        for i, (model, score_data) in enumerate(sorted_models, 1):
            if isinstance(score_data, dict):
                score = score_data['score']
                ci = score_data['confidence_interval']
                summary_lines.append(f"{i}. **{model}**: {score:.2f}% ± {ci['margin_of_error']:.2f}%")
            else:
                summary_lines.append(f"{i}. **{model}**: {score_data:.2f}%")
        
        # Category comparison
        summary_lines.append("\n## Performance by Category")
        categories = sorted(comparison['category_comparison'].keys())
        
        for category in categories:
            scores = comparison['category_comparison'][category]
            sorted_scores = sorted(scores.items(), 
                                 key=lambda x: x[1]['score'] if isinstance(x[1], dict) else x[1], 
                                 reverse=True)
            
            summary_lines.append(f"\n### {category}")
            for model, score_data in sorted_scores:
                if isinstance(score_data, dict):
                    score = score_data['score']
                    summary_lines.append(f"- {model}: {score:.2f}%")
                else:
                    summary_lines.append(f"- {model}: {score_data:.2f}%")
        
        # Error rate comparison
        summary_lines.append("\n## Error Rates")
        sorted_errors = sorted(comparison['error_rate_comparison'].items(), 
                             key=lambda x: x[1])
        
        for model, error_rate in sorted_errors:
            summary_lines.append(f"- {model}: {error_rate:.2f}%")
        
        # Save summary
        summary_file = output_dir / 'model_comparison_summary.md'
        with open(summary_file, 'w') as f:
            f.write('\n'.join(summary_lines))


from collections import defaultdict