# CodeEval Dataset

The official CodeEval benchmark dataset for evaluating Large Language Models on Python code generation tasks.

## Overview

CodeEval is a pedagogical benchmark dataset designed to evaluate code-trained LLMs across multiple Python programming categories with varying complexity levels. The dataset contains programming problems with:

- **24+ categories**: dataclass, typing, decorator, inheritance, context managers, and more
- **3 complexity levels**: Simple, medium, and advanced problems
- **Multiple problem types**: Function and class implementations
- **Comprehensive test cases**: Each problem includes assertion-based tests

## Quick Start

### Download the Dataset

```bash
# From the dataset directory
cd dataset

# Download and extract the dataset (default: generated/ folder)
python download.py

# Download to a specific directory
python download.py --output-dir /path/to/data

# Download without extracting
python download.py --no-extract
```

### Verify Download

```bash
# Verify MD5 checksum of existing download
python download.py --verify-only
```

## Download Options

The `download.py` script supports several options:

```bash
python download.py [OPTIONS]

Options:
  --output-dir DIR    Directory to save the dataset (default: generated/)
  --no-extract        Download the zip file without extracting
  --verify-only       Only verify existing download without downloading again
  --help              Show help message
```

### Examples

```bash
# Standard download and extract
python download.py

# Download to custom location
python download.py --output-dir ~/datasets/codeeval

# Just download the zip without extracting
python download.py --no-extract

# Verify integrity of previously downloaded file
python download.py --verify-only
```

## Dataset Format

After extraction, you'll find JSONL files organized by category in the `generated/` folder:

```
dataset/generated/
├── dataclass.jsonl      # 25 dataclass problems
├── typing.jsonl         # 23 typing problems
├── decorator.jsonl      # 28 decorator problems
├── inheritance.jsonl    # Problems on class inheritance
└── ...                  # 24 categories total
```

### Problem Structure

Each line in the JSONL files contains one problem:

```json
{
  "task_id": "dataclass_1",
  "name": "create_person_class",
  "problem": "Create a dataclass for a Person with name and age fields",
  "canonical_solution": "@dataclass\nclass Person:\n    name: str\n    age: int",
  "tests": "[{\"ctx\": \"p = Person('Alice', 30)\", \"assertion\": \"p.name == 'Alice' and p.age == 30\"}]",
  "topic": "dataclass",
  "object": "class",
  "complexity": 2
}
```

**Field Descriptions:**
- `task_id`: Unique identifier for the problem
- `name`: Descriptive name of the problem
- `problem`: Problem statement/description
- `canonical_solution`: Reference solution
- `tests`: JSON array of test cases with context and assertions
- `topic`: Category/topic of the problem
- `object`: Type of implementation required (function or class)
- `complexity`: Difficulty level (1=simple, 2=medium, 3=advanced)

## Using the Dataset

### With runcodeeval

Once downloaded, use the dataset with the runcodeeval evaluation framework:

```bash
# Navigate to runcodeeval directory
cd ../runcodeeval

# Evaluate LLM solutions
python main.py --benchmark ../dataset/generated --solutions /path/to/solutions
```

### With llm_solutions

Generate solutions for the dataset using llm_solutions:

```bash
# Navigate to llm_solutions directory
cd ../llm_solutions

# Generate solutions for the dataset
# (See llm_solutions/README.md for specific instructions)
```

## Dataset Statistics

- **Total Problems**: 600+ programming challenges
- **Categories**: 24 different Python topics
- **Complexity Levels**: 3 levels (simple, medium, advanced)
- **Problem Types**: Functions and classes
- **File Size**: ~108 KB (compressed)

## Dataset Metadata

The dataset includes a `croissant.json` file following the [Croissant metadata format](https://github.com/mlcommons/croissant), which provides:

- Standardized dataset description
- Download URLs and checksums
- Creator and license information
- Citation information

## Citation

If you use the CodeEval dataset in your research, please see the [main README](../README.md#citation) for citation information.

## License

The CodeEval dataset is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to:
- **Share**: Copy and redistribute the material
- **Adapt**: Remix, transform, and build upon the material

Under the following terms:
- **Attribution**: You must give appropriate credit

## Dataset Source

- **DOI**: [10.5281/zenodo.17495202](https://doi.org/10.5281/zenodo.17495202)
- **Publisher**: Zenodo
- **Creators**: Danny Brahman, Dr. Mohammad Mahoor (University of Denver)
- **Version**: 1.0.0
- **Published**: January 2025

## Troubleshooting

### Download Issues

**Connection timeout or slow download:**
- Try downloading during off-peak hours
- Use `--no-extract` to download the zip only, then extract manually

**MD5 checksum mismatch:**
- The downloaded file may be corrupted
- Delete the zip file and try downloading again
- Check your internet connection

**Permission denied:**
- Ensure you have write permissions for the output directory
- Try specifying a different output directory with `--output-dir`

### Extraction Issues

**Extraction fails:**
- Verify the download completed successfully with `--verify-only`
- Try extracting manually: `unzip codeeval.zip`

### File Format Issues

**JSONL parsing errors:**
- Ensure files are not corrupted during download
- Each line should be a valid JSON object
- Use a JSON validator to check file integrity

## Support

For issues or questions:
- Check the [main repository README](../README.md)
- Review the [runcodeeval documentation](../runcodeeval/README.md)
- Open an issue on GitHub

## Related Components

- **[runcodeeval](../runcodeeval/)**: Evaluation framework for LLM solutions
- **[llm_solutions](../llm_solutions/)**: Solution generator for obtaining LLM outputs
