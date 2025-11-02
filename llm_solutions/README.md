# LLM Solutions Collection Project

This file provides guidance to developers when working with the LLM solutions collection project.

## Overview

The LLM solutions project collects Python code solutions from multiple LLM providers in parallel. It processes the dataset problems, prompts various LLMs, extracts clean Python code, and stores results for analysis.

## Project Structure

```
llm_solutions/
├── scripts/               # Main execution scripts
│   ├── run_collection.py  # Primary collection script
│   └── resume_collection.py # Resume interrupted sessions
├── config/               # Configuration and settings
│   ├── model_configs.py  # LLM provider configurations
│   └── collection_settings.py # Collection parameters
├── models/               # LLM provider implementations
│   ├── base_model.py     # Abstract base class
│   ├── openai_model.py   # GPT models
│   ├── anthropic_model.py # Claude models
│   ├── google_model.py   # Gemini models
│   ├── xai_model.py      # Grok models
│   ├── meta_model.py     # Llama models (via Together AI & HuggingFace)
│   └── deepseek_model.py # DeepSeek models
├── processors/           # Response processing
│   ├── solution_extractor.py # Agentic code extraction
│   └── response_validator.py # Solution validation
├── runners/              # Execution orchestration
│   ├── collection_runner.py # Main collection orchestrator
│   ├── parallel_executor.py # Parallel task execution
│   └── rate_limiter.py   # Adaptive rate limiting
├── storage/              # Data persistence
│   ├── solution_store.py # Solution storage and export
│   └── progress_tracker.py # Session progress tracking
├── prompts/              # Prompt templates
│   ├── prompt_template.py # Problem-to-prompt conversion
│   └── system_prompts.py # System message templates
├── .env.example          # API key template
├── requirements.txt      # Python dependencies
└── generated/            # Output directory for collected solutions
    ├── session_1234567890/  # Session-based organization
    │   ├── gpt-4/          # Model-specific folders
    │   │   ├── dataclass.jsonl      # Category-specific solutions
    │   │   ├── typing.jsonl
    │   │   └── summary.json         # Model performance summary
    │   ├── claude-3-sonnet/
    │   │   ├── dataclass.jsonl
    │   │   ├── typing.jsonl
    │   │   └── summary.json
    │   └── cross_model_comparison.json # Cross-model comparison
    └── session_1234567891/
        └── ...
```

## Key Features

### Multi-Provider LLM Support
- **OpenAI**: GPT-3.5, GPT-4 series models
- **Anthropic**: Claude 3 models (Haiku, Sonnet, Opus)
- **Google**: Gemini models
- **xAI**: Grok models
- **Meta**: Llama models via Together AI and HuggingFace
- **DeepSeek**: DeepSeek-Coder models

### Session-Based Organization
- **Complete isolation**: Each collection run gets its own session folder
- **No overwrites**: Previous results are always preserved
- **Category organization**: Each model's solutions organized by category (matching dataset structure)
- **Easy comparison**: Can compare results across different sessions and models

### Parallel Processing
- **Concurrent execution**: Multiple models process problems simultaneously
- **Rate limiting**: Adaptive rate limiting per provider
- **Batch processing**: Problems processed in configurable batches
- **Session management**: Resume interrupted collections

### Solution Extraction
- **Agentic approach**: Uses LLM to extract clean Python code from responses
- **Fallback extraction**: Basic regex-based extraction when no API key provided
- **Validation**: Syntax and structure validation of extracted code

## Setup and Configuration

### Environment Setup
```bash
cd llm_solutions

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

### API Keys Configuration
Required environment variables in `.env`:
```bash
# LLM Provider API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
XAI_API_KEY=your_xai_api_key_here
TOGETHER_API_KEY=your_together_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# OpenAI Organization and Project (required for project-scoped keys)
OPENAI_ORG_ID=org-xxxxxxxxxxxxxxxxxx
OPENAI_PROJECT_ID=proj_xxxxxxxxxxxxxxxxxx

# Collection Settings (optional)
DATASET_PATH=../dataset/generated/codeeval
OUTPUT_DIR=generated
BATCH_SIZE=10
MAX_CONCURRENT_PER_MODEL=3
```

## Usage

### Checking Supported Models

**Before running collection**, check which models are currently supported in the codebase:

```bash
# List all models implemented in the code with their API key status
python scripts/run_collection.py --list-models
```

**Example output:**
```
Available models:
  gpt_4 (openai) ✓
  gpt_3_5_turbo (openai) ✓
  claude-4-sonnet (anthropic) ✓
  gemini_2_0_flash (google) ✓
  ...

API Key Status:
  OPENAI_API_KEY: ✓
  ANTHROPIC_API_KEY: ✗
  ...
```

- Models shown are **implemented in the codebase**
- `✓` next to a model means its API key is configured in `.env`
- `✗` means the API key is missing (add it to `.env` to use that model)

**If your desired model is not in this list**, you need to add support for it by implementing a new model class following the architecture (see "Adding New Models" section below).

### Basic Collection
```bash
# Use specific models (must be exact names from --list-models)
python scripts/run_collection.py --models gpt_4 claude-4-sonnet

# Use all supported models with configured API keys
python scripts/run_collection.py --all-models

# Quick test with 1 problem per category
python scripts/run_collection.py --quick-test
```

### Advanced Options
```bash
# Target specific categories
python scripts/run_collection.py --categories typing inheritance

# Custom batch and concurrency settings
python scripts/run_collection.py --batch-size 20 --max-concurrent 5

# Save detailed logs
python scripts/run_collection.py --log-level DEBUG --log-file collection.log
```

### Testing and Validation Options

#### Dry Run (`--dry-run`)
Validates the entire setup without making any LLM API calls:
```bash
# Validate setup without running collection
python scripts/run_collection.py --dry-run
python scripts/run_collection.py --dry-run --models gpt-4 claude-3-sonnet
```

**What `--dry-run` does:**
- ✅ Checks API key availability for specified models
- ✅ Initializes and validates model configurations
- ✅ Loads and counts problems from dataset
- ✅ Creates collection runner and validates settings
- ✅ Reports total tasks that would be executed
- ❌ **Does NOT call LLMs or collect solutions**
- ✅ Returns success if everything validates correctly

**Example output:**
```
INFO - Initializing models...
INFO - Initialized 2 models: ['gpt-4', 'claude-3-sonnet']
INFO - Loading dataset...
INFO - Loaded 602 problems
INFO - Total tasks to complete: 1204
INFO - Dry run completed successfully
```

#### Quick Test (`--quick-test`)
Runs a real collection with minimal problems for testing:
```bash
# Test with 1 problem per category (24 problems total)
python scripts/run_collection.py --quick-test

# Test specific categories only
python scripts/run_collection.py --quick-test --categories dataclass typing

# Test with specific models
python scripts/run_collection.py --quick-test --models gpt-3.5-turbo
```

**What `--quick-test` does:**
- ✅ **Actually calls LLMs** and collects real solutions
- ✅ Processes only **1 problem per category** instead of all problems
- ✅ Uses reduced batch sizes and concurrency for testing
- ✅ Saves results in the same session-based structure

**Quick Test Problem Selection:**
- **No categories specified**: 1 problem from each of 24 categories = 24 problems
- **Specific categories**: 1 problem from each specified category
- **Example**: `--categories dataclass typing` = 2 problems total

**Quick Test with Default Models:**
If no models specified, uses: `["gpt-3.5-turbo", "claude-3-haiku-20240307"]`

**LLM Call Examples:**
```bash
# 24 categories × 2 default models = 48 LLM calls
python scripts/run_collection.py --quick-test

# 2 categories × 2 default models = 4 LLM calls  
python scripts/run_collection.py --quick-test --categories dataclass typing

# 24 categories × 1 model = 24 LLM calls
python scripts/run_collection.py --quick-test --models gpt-4

# 1 category × 1 model = 1 LLM call
python scripts/run_collection.py --quick-test --categories dataclass --models gpt-4
```

### Session Management
```bash
# List available sessions
python scripts/resume_collection.py --list-sessions

# Show session status
python scripts/resume_collection.py session_1234567890 --status

# Resume interrupted session using original parameters
python scripts/resume_collection.py session_1234567890
```

#### Resume Functionality
The resume feature automatically uses original session parameters and skips completed problems:

**How Resume Works:**
1. **Load Session Parameters**: Automatically restores original models, categories, quick_test mode, dataset path, and other settings from the session's summary.json
2. **Restore Progress**: Loads progress tracking from the previous session
3. **Filter Completed**: Automatically excludes already-completed problems from processing
4. **Continue Processing**: Resumes with original parameters, processing only remaining problems

**Key Benefits:**
- **Pure Resume**: No need to remember or specify original parameters (models, categories, quick_test, etc.)
- **Cost Savings**: Prevents duplicate LLM API calls by skipping completed problems
- **Session Isolation**: Each resume uses exact original parameters, ensuring consistency

**Simple Usage:**
```bash
# Original session
python scripts/run_collection.py --models gpt-4 claude-3-sonnet --categories dataclass typing --quick-test

# Resume (automatically uses: gpt-4, claude-3-sonnet, dataclass+typing, quick_test=true)
python scripts/resume_collection.py session_1234567890
```

**Session Parameter Storage**: All original parameters are stored in the session-level summary.json file under `collection_metadata.session_parameters`, making resume completely autonomous.

## Output Organization

### Session Structure
Each collection creates a session folder under `generated/`:
```
generated/
├── session_1234567890/                   # Session folder
│   ├── progress.json                    # Real-time progress tracking and resume state
│   ├── final_report.json                # Final session summary
│   ├── summary.json                     # Combined session & model summaries
│   ├── gpt-4/                           # Model folder (solutions only)
│   │   ├── dataclass.jsonl              # Solutions for dataclass category
│   │   ├── dataclass.csv                # CSV format solutions
│   │   ├── typing.jsonl                 # Solutions for typing category
│   │   └── typing.csv                   # CSV format solutions
│   ├── claude-3-sonnet/
│   │   ├── dataclass.jsonl
│   │   ├── dataclass.csv
│   │   ├── typing.jsonl
│   │   └── typing.csv
│   └── cross_model_comparison.json      # Cross-model performance comparison
└── session_1234567891/                  # Another session
    └── ...
```

### Progress Tracking Files

#### Progress File (`progress.json`)
Real-time progress tracking and resume state management:
```json
{
  "session_id": "session_1234567890",
  "start_time": 1234567890.123,
  "total_tasks": 24,
  "models": ["gpt-4", "claude-3-sonnet"],
  "categories": ["dataclass", "typing"],
  "status": "in_progress",
  "current_progress": {
    "completed_tasks": 10,
    "total_tasks": 24,
    "successful_tasks": 9,
    "failed_tasks": 1,
    "model_progress": {
      "gpt-4": {
        "completed": 5,
        "successful": 5,
        "failed": 0,
        "avg_time": 2.1,
        "last_error": null
      }
    },
    "completed_task_ids": [
      "dataclass_1_gpt-4",
      "dataclass_2_gpt-4"
    ],
    "estimated_completion": 1234568000.123
  },
  "last_saved": 1234567900.456
}
```
- **Purpose**: Monitor progress and enable session resume
- **Updates**: After each task completion + auto-save every 30 seconds
- **Status Values**: `in_progress`, `completed`, `failed`
- **Resume Support**: Contains all state needed to resume interrupted sessions
- **Task IDs**: Tracks completed tasks to avoid re-processing during resume

#### Final Report (`final_report.json`)
Comprehensive summary created at session completion:
```json
{
  "session_id": "session_1234567890",
  "progress": {
    "completed": 24,
    "total": 24,
    "successful": 22,
    "failed": 2,
    "progress_percent": 100.0,
    "success_rate": 91.7
  },
  "timing": {
    "elapsed_time": 145.2,
    "estimated_remaining": 0,
    "estimated_total": 145.2,
    "last_update": 1234568035.123
  },
  "model_progress": {
    "gpt-4": {
      "completed": 12,
      "successful": 11,
      "failed": 1,
      "avg_time": 2.3,
      "last_error": "Rate limit exceeded"
    }
  },
  "status": "completed",
  "end_time": 1234568035.456,
  "total_duration": 145.2
}
```
- **Purpose**: Final collection summary and statistics
- **Updates**: Once at session completion
- **Use**: Overall performance analysis, success metrics

#### Progress Tracking Summary

| File Type | Update Frequency | Purpose | Key Data |
|-----------|-----------------|---------|----------|
| `progress.json` | Continuous (real-time) | Live monitoring & resume | Current state, task counts, completed task IDs, status |
| `final_report.json` | Once at completion | Final summary | Complete statistics, duration |
| `summary.json` | Once at completion | Session & model summaries | All model performances, parameters |

### File Formats

#### Category Solutions (`model/category.jsonl`)
Each line contains a solution record:
```json
{
  "task_id": "dataclass_1",
  "model": "gpt-4",
  "problem": "Create a dataclass...",
  "category": "dataclass",
  "complexity": 2,
  "raw_response": "Here's the solution:\n\n```python\n...",
  "extracted_solution": "from dataclasses import dataclass...",
  "validation": {
    "is_fully_valid": true,
    "syntax": {"is_valid": true, "error": null},
    "structure": {"has_expected_name": true, "found_names": ["SimpleData"]},
    "imports": {"has_imports": true, "from_imports": ["dataclasses.dataclass"]},
    "code_length": 90,
    "line_count": 6
  },
  "llm_call_success": true,
  "execution_time": 2.3,
  "timestamp": 1234567890.123
}
```

#### Session Summary (`summary.json`)
Combined session metadata and model performance statistics:
```json
{
  "session_id": "session_1234567890",
  "collection_metadata": {
    "collection_start": 1234567890.123,
    "collection_end": 1234568035.456,
    "total_problems": 24,
    "models_evaluated": ["gpt-4", "claude-3-sonnet"],
    "categories_included": ["dataclass", "typing"],
    "session_id": "session_1234567890",
    "session_parameters": {
      "models": ["gpt-4", "claude-3-sonnet"],
      "categories": ["dataclass", "typing"],
      "quick_test": true,
      "dataset_path": "../dataset/generated",
      "batch_size": 10,
      "max_concurrent_per_model": 3,
      "output_dir": "generated"
    }
  },
  "model_summaries": {
    "gpt-4": {
      "model_name": "gpt-4",
      "overview": {
        "total_problems": 24,
        "total_solutions": 24,
        "categories_processed": ["dataclass", "typing"]
      },
      "overall_performance": {
        "success_rate": 95.5,
        "syntax_validity_rate": 98.2,
        "avg_execution_time": 2.1
      },
      "category_performance": {
        "dataclass": {"success_rate": 97.0, "syntax_validity_rate": 100.0},
        "typing": {"success_rate": 92.1, "syntax_validity_rate": 96.8}
      }
    },
    "claude-3-sonnet": {
      "model_name": "claude-3-sonnet",
      "overview": {...},
      "overall_performance": {...},
      "category_performance": {...}
    }
  },
  "collection_date": "2025-07-18 14:54:04"
}
```

#### Cross-Model Comparison (`cross_model_comparison.json`)
Session-level comparison across all models:
```json
{
  "summary": {
    "session_id": "session_1234567890",
    "models_evaluated": ["gpt-4", "claude-3-sonnet"],
    "categories_processed": ["dataclass", "typing", "decorator"],
    "total_problems": 150
  },
  "problems": [
    {
      "task_id": "dataclass_1",
      "problem": "Create a dataclass...",
      "category": "dataclass",
      "gpt-4_valid": true,
      "gpt-4_time": 2.1,
      "claude-3-sonnet_valid": true,
      "claude-3-sonnet_time": 1.8
    }
  ]
}
```

## Architecture Components

### Solution Storage (`storage/`)
The `SolutionStore` class manages the session-based organization:

```python
store = SolutionStore(output_dir="generated", session_id="session_1234567890")

# Automatically creates: generated/session_1234567890/
# Add solutions (automatically grouped by model and category)
store.add_single_solution(task_id, model_name, solution_data)

# Save in organized structure
jsonl_paths = store.save_solutions_jsonl()  # Returns {model: {category: path}}
csv_paths = store.save_solutions_csv()      # Returns {model: {category: path}}
summary_paths = store.save_analysis_summary()  # Returns {model: path}
comparison_path = store.save_cross_model_comparison()  # Returns path
```

### Solution Extraction (`processors/`)
Two extraction approaches:

1. **Agentic Extraction**: Uses LLM to clean responses
   ```python
   extractor = SolutionExtractor(extraction_api_key)
   clean_code = await extractor.extract_python_code(raw_response, task_id)
   ```

2. **Fallback Extraction**: Regex-based code block extraction
   ```python
   # Finds ```python...``` blocks
   pattern = r'```(?:python|py)?\s*\n?(.*?)\n?```'
   ```

## Adding New Models

If your desired model is not supported, you can add it by following the existing architecture:

### Step 1: Create a Model Implementation

Create a new file in `models/` (e.g., `models/my_provider_model.py`) extending `BaseModel`:

```python
from models.base_model import BaseModel
import aiohttp

class MyProviderModel(BaseModel):
    """Implementation for MyProvider models"""

    async def generate_completion(self, prompt: str) -> str:
        """Generate completion using MyProvider API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.myprovider.com/v1/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                result = await response.json()
                return result['choices'][0]['message']['content']
```

### Step 2: Register in Model Configs

Add your model to `config/model_configs.py`:

```python
ModelConfig(
    name="my_model_name",
    provider="myprovider",
    api_key_env="MYPROVIDER_API_KEY",
    model_class="MyProviderModel",
    model_params={
        "model_name": "my-model-v1",
        "max_tokens": 2048,
        "temperature": 0.7
    }
)
```

### Step 3: Add API Key to .env

```bash
MYPROVIDER_API_KEY=your_api_key_here
```

### Step 4: Verify Setup

```bash
# Check if your model appears in the list
python scripts/run_collection.py --list-models

# Test with dry-run
python scripts/run_collection.py --models my_model_name --dry-run

# Run quick test
python scripts/run_collection.py --models my_model_name --quick-test
```

Refer to existing model implementations in `models/` (e.g., `openai_model.py`, `anthropic_model.py`) for complete examples.

## Development Guidelines

### Working with Sessions
- Each session is completely isolated
- Session IDs are timestamp-based by default
- Can specify custom session IDs for testing
- Previous sessions are never overwritten

### Category-Based Organization
- Solutions automatically organized by category (matching dataset structure)
- Easy to compare model performance on specific topics
- Can load specific categories: `store.load_solutions_from_jsonl(model, category)`
- Get available categories: `store.get_category_files(model)`

## Integration with Dataset Project

The LLM solutions project integrates with the dataset:
- Reads problems from `../dataset/generated/*.jsonl`
- Mirrors the category-based organization structure  
- Processes problems using `func`/`cls` conventions
- Outputs solutions in parallel structure for easy comparison

### Expected Problem Format
```json
{
  "task_id": "category_N",
  "name": "function_or_class_name",
  "problem": "Problem description using 'func' or 'cls'",
  "canonical_solution": "reference_implementation",
  "tests": "[{\"ctx\": \"\", \"assertion\": \"...\"}]",
  "category": "category_name",
  "complexity": 2,
  "object": "function"
}
```