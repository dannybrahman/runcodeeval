#!/usr/bin/env python3
"""
CodeEval Dataset Downloader

This script downloads the CodeEval benchmark dataset from Zenodo
using the metadata defined in croissant.json.

Usage:
    python download.py [--output-dir DIR] [--no-extract] [--verify-only]

Options:
    --output-dir DIR    Directory to save the dataset (default: generated/)
    --no-extract        Download the zip file without extracting
    --verify-only       Only verify existing download without downloading again
    --help              Show this help message
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path


def load_croissant_metadata(croissant_path: str = "croissant.json") -> dict:
    """Load metadata from croissant.json file."""
    try:
        with open(croissant_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: croissant.json not found at {croissant_path}")
        print("Make sure you're running this script from the dataset/ directory")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in croissant.json: {e}")
        sys.exit(1)


def calculate_md5(filepath: str, chunk_size: int = 8192) -> str:
    """Calculate MD5 checksum of a file."""
    md5_hash = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def download_file(url: str, output_path: str, expected_md5: str = None) -> bool:
    """Download file with progress bar and optional MD5 verification."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    try:
        # Download with progress
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, output_path, reporthook=reporthook)
        print("\nDownload completed!")

        # Verify MD5 if provided
        if expected_md5:
            print("Verifying file integrity...")
            actual_md5 = calculate_md5(output_path)
            if actual_md5 == expected_md5:
                print(f"✓ MD5 checksum verified: {actual_md5}")
                return True
            else:
                print(f"✗ MD5 checksum mismatch!")
                print(f"  Expected: {expected_md5}")
                print(f"  Got:      {actual_md5}")
                print("  File may be corrupted. Please try downloading again.")
                return False
        return True

    except urllib.error.URLError as e:
        print(f"\nError downloading file: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False


def extract_zip(zip_path: str, extract_dir: str) -> bool:
    """Extract zip file to specified directory."""
    print(f"\nExtracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            extracted_files = zip_ref.namelist()
            print(f"✓ Extracted {len(extracted_files)} files to {extract_dir}")
            print("\nExtracted files:")
            for file in extracted_files[:10]:  # Show first 10 files
                print(f"  - {file}")
            if len(extracted_files) > 10:
                print(f"  ... and {len(extracted_files) - 10} more files")
            return True
    except zipfile.BadZipFile:
        print(f"✗ Error: {zip_path} is not a valid zip file")
        return False
    except Exception as e:
        print(f"✗ Error extracting file: {e}")
        return False


def verify_existing_download(filepath: str, expected_md5: str) -> bool:
    """Verify an existing downloaded file."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False

    print(f"Verifying {filepath}...")
    actual_md5 = calculate_md5(filepath)

    if actual_md5 == expected_md5:
        print(f"✓ MD5 checksum verified: {actual_md5}")
        return True
    else:
        print(f"✗ MD5 checksum mismatch!")
        print(f"  Expected: {expected_md5}")
        print(f"  Got:      {actual_md5}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download CodeEval dataset from Zenodo using croissant.json metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='generated',
        help='Directory to save the dataset (default: generated/)'
    )
    parser.add_argument(
        '--no-extract',
        action='store_true',
        help='Download the zip file without extracting'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing download without downloading again'
    )

    args = parser.parse_args()

    # Load metadata
    print("Loading metadata from croissant.json...")
    metadata = load_croissant_metadata()

    # Extract download information
    dataset_name = metadata.get('name', 'CodeEval')
    version = metadata.get('version', '1.0.0')
    print(f"Dataset: {dataset_name} v{version}")
    print(f"Description: {metadata.get('description', 'N/A')[:100]}...")

    distribution = metadata.get('distribution', [])
    if not distribution:
        print("Error: No distribution information found in croissant.json")
        sys.exit(1)

    # Get the first distribution (the zip file)
    file_info = distribution[0]
    download_url = file_info.get('contentUrl')
    filename = file_info.get('name', 'codeeval.zip')
    expected_md5 = file_info.get('md5')

    if not download_url:
        print("Error: No download URL found in croissant.json")
        sys.exit(1)

    # Set up output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / filename

    # Verify only mode
    if args.verify_only:
        success = verify_existing_download(str(zip_path), expected_md5)
        sys.exit(0 if success else 1)

    # Check if file already exists
    if zip_path.exists():
        print(f"\n{filename} already exists.")
        response = input("Do you want to re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping download.")
            if not args.no_extract:
                extract_success = extract_zip(str(zip_path), str(output_dir))
                sys.exit(0 if extract_success else 1)
            sys.exit(0)

    # Download the file
    print(f"\n{'='*60}")
    print(f"Downloading {dataset_name}")
    print(f"{'='*60}\n")

    success = download_file(download_url, str(zip_path), expected_md5)

    if not success:
        print("\nDownload failed!")
        sys.exit(1)

    # Extract if requested
    if not args.no_extract:
        extract_success = extract_zip(str(zip_path), str(output_dir))
        if not extract_success:
            sys.exit(1)

    print(f"\n{'='*60}")
    print("✓ CodeEval dataset ready!")
    print(f"{'='*60}")
    print(f"\nLocation: {output_dir.absolute()}")
    if not args.no_extract:
        print("\nYou can now use this dataset with runcodeeval:")
        print(f"  cd ../runcodeeval")
        print(f"  python main.py --benchmark ../dataset/generated --solutions <your-solutions>")
    print("\nFor more information, see the README.md")


if __name__ == '__main__':
    main()
