# CodeEval: LLM Code Generation Benchmark

This repository provides tools for evaluating Large Language Models (LLMs) on the open-source CodeEval benchmark dataset.

## Repository Structure

This repository contains three main components:

### 1. **dataset** - CodeEval Benchmark Dataset
Located in `dataset/`, this folder provides tools to download the official CodeEval benchmark dataset from Zenodo. The dataset contains 600+ Python programming problems across 24 categories with varying complexity levels.

[📖 View dataset documentation](dataset/README.md)

**Key features:**
- Easy download script using croissant.json metadata
- Automatic MD5 checksum verification
- 24+ Python programming categories
- 3 complexity levels per category
- Comprehensive test cases for each problem

### 2. **llm_solutions** - Solution Generator (Optional)
Located in `llm_solutions/`, this **optional** tool generates LLM solutions for CodeEval benchmark problems. You can use this tool to obtain solutions from various LLM models, or use your own solution generation method as long as it produces solutions in the [required format](runcodeeval/README.md#solution-data).

[📖 View llm_solutions documentation](llm_solutions/README.md)

**Key features:**
- Generate solutions from multiple LLM providers
- Support for various models (GPT-4, Claude, etc.)
- Configurable prompting strategies
- Batch processing capabilities

### 3. **runcodeeval** - Evaluation Framework
Located in `runcodeeval/`, this software evaluates LLM-generated solutions against the CodeEval benchmark dataset. It produces detailed metrics including overall scores, category-based performance, complexity analysis, and error reporting.

[📖 View runcodeeval documentation](runcodeeval/README.md)

**Key features:**
- Comprehensive evaluation with detailed metrics
- Category and complexity breakdowns
- Error analysis and reporting
- Multi-model comparison
- Exam-based scoring system

## Quick Start Workflow

### Complete Evaluation Pipeline

1. **Download the dataset**:
   ```bash
   cd dataset
   python download.py
   cd ..
   ```

2. **Generate solutions** (optional - use llm_solutions or your own method):
   ```bash
   cd llm_solutions
   # Follow instructions in llm_solutions/README.md
   cd ..
   ```

   **Or** use your own solution generation method that outputs JSONL files with the [required format](runcodeeval/README.md#solution-data).

3. **Evaluate solutions**:
   ```bash
   cd runcodeeval
   python main.py --benchmark ../dataset/generated --solutions <path-to-your-solutions>
   ```

## What is CodeEval?

CodeEval is an open-source benchmark dataset for evaluating LLMs on programming tasks. It contains problems across multiple categories (dataclass, typing, decorator, inheritance, etc.) with varying complexity levels.

## Citation

If you use CodeEval in your research, please cite our work:

```bibtex
@inproceedings{brahman-mahoor-2025-codeeval,
    title = "{C}ode{E}val: A pedagogical approach for targeted evaluation of code-trained Large Language Models",
    author = "Brahman, Danny and Mahoor, Mohammad",
    booktitle = "Proceedings of the International Joint Conference on Natural Language Processing {\&} Asia-Pacific Chapter of the Association for Computational Linguistics",
    year = "2025",
    publisher = "Association for Computational Linguistics",
    note = "Accepted - To appear"
}
```

## License

This project is licensed under the terms in the [LICENSE](LICENSE) file.

## Contributing

Contributions are welcome! Please refer to the individual component READMEs for specific contribution guidelines.
