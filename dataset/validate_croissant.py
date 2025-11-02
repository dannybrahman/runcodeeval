#!/usr/bin/env python3
"""
Croissant Metadata Validator

This script validates the croissant.json file to ensure it conforms
to the Croissant metadata format specification.

Usage:
    python validate_croissant.py [--file croissant.json]

The script will:
1. Check if mlcroissant is installed (and provide installation instructions if not)
2. Validate the JSON structure
3. Validate against Croissant schema
4. Report any errors or warnings
"""

import argparse
import json
import sys
from pathlib import Path


def validate_json_structure(filepath: str) -> bool:
    """Validate basic JSON structure."""
    print("Step 1: Validating JSON structure...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("✓ Valid JSON structure")
        return True, data
    except FileNotFoundError:
        print(f"✗ Error: File not found: {filepath}")
        return False, None
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON - {e}")
        return False, None


def check_required_fields(data: dict) -> bool:
    """Check for required Croissant fields."""
    print("\nStep 2: Checking required fields...")

    required_fields = {
        "@context": "Context definition",
        "@type": "Resource type",
        "name": "Dataset name",
        "description": "Dataset description",
        "url": "Dataset URL",
        "distribution": "Distribution information"
    }

    all_present = True
    for field, description in required_fields.items():
        if field in data:
            print(f"✓ {description} ({field}): present")
        else:
            print(f"✗ {description} ({field}): MISSING")
            all_present = False

    return all_present


def validate_with_mlcroissant(filepath: str) -> bool:
    """Validate using mlcroissant library if available."""
    print("\nStep 3: Validating with mlcroissant library...")

    try:
        import mlcroissant
        print("✓ mlcroissant library found")

        # Try to load and validate
        try:
            # The mlcroissant library can validate files
            import subprocess
            result = subprocess.run(
                ['mlcroissant', 'validate', '--jsonld', filepath],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print("✓ Croissant validation passed!")
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print("Warnings:", result.stderr)
                return True
            else:
                print("✗ Croissant validation failed!")
                print(result.stdout)
                print(result.stderr)
                return False

        except Exception as e:
            print(f"Note: Could not run mlcroissant CLI: {e}")
            print("You can manually validate with: mlcroissant validate --jsonld croissant.json")
            return None

    except ImportError:
        print("⚠ mlcroissant library not installed")
        print("\nTo install:")
        print("  pip install mlcroissant")
        print("\nOr validate manually:")
        print("  pip install mlcroissant")
        print(f"  mlcroissant validate --jsonld {filepath}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Validate Croissant metadata file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--file',
        type=str,
        default='croissant.json',
        help='Path to croissant.json file (default: croissant.json)'
    )

    args = parser.parse_args()
    filepath = args.file

    print(f"{'='*60}")
    print(f"Validating Croissant Metadata: {filepath}")
    print(f"{'='*60}\n")

    # Step 1: Validate JSON structure
    json_valid, data = validate_json_structure(filepath)
    if not json_valid:
        sys.exit(1)

    # Step 2: Check required fields
    fields_valid = check_required_fields(data)

    # Step 3: Validate with mlcroissant if available
    mlcroissant_result = validate_with_mlcroissant(filepath)

    # Summary
    print(f"\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")

    if json_valid and fields_valid:
        if mlcroissant_result is True:
            print("✓ All validations passed!")
            print("Your croissant.json file is valid and ready to use.")
            sys.exit(0)
        elif mlcroissant_result is None:
            print("⚠ Basic validation passed")
            print("Install mlcroissant for complete validation:")
            print("  pip install mlcroissant")
            sys.exit(0)
        else:
            print("✗ Croissant schema validation failed")
            print("Please fix the errors above")
            sys.exit(1)
    else:
        print("✗ Validation failed")
        print("Please fix the errors above")
        sys.exit(1)


if __name__ == '__main__':
    main()
