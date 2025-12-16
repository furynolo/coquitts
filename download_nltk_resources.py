#!/usr/bin/env python3
"""
Script to download required NLTK resources for the book-to-audiobook converter.
"""

import nltk
import sys

def download_nltk_resources():
    """Download all required NLTK resources."""
    print("Downloading NLTK resources...")
    print("=" * 50)
    
    resources = [
        ('punkt', 'tokenizers/punkt'),
        ('punkt_tab', 'tokenizers/punkt_tab'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
    ]
    
    for resource_name, resource_path in resources:
        print(f"\nChecking {resource_name}...")
        try:
            nltk.data.find(resource_path)
            print(f"  [OK] {resource_name} already available")
        except LookupError:
            print(f"  Downloading {resource_name}...")
            try:
                nltk.download(resource_name, quiet=False)
                # Verify it was downloaded
                try:
                    nltk.data.find(resource_path)
                    print(f"  [OK] {resource_name} downloaded successfully")
                except LookupError:
                    print(f"  [WARNING] {resource_name} download may have failed")
            except Exception as e:
                print(f"  [ERROR] Error downloading {resource_name}: {e}")
    
    print("\n" + "=" * 50)
    print("NLTK resource download complete!")
    print("\nVerifying all resources...")
    
    all_ok = True
    for resource_name, resource_path in resources:
        try:
            nltk.data.find(resource_path)
            print(f"  [OK] {resource_name} is available")
        except LookupError:
            print(f"  [MISSING] {resource_name} is NOT available")
            all_ok = False
    
    if all_ok:
        print("\n[SUCCESS] All NLTK resources are available!")
        return 0
    else:
        print("\n[ERROR] Some NLTK resources are missing. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(download_nltk_resources())

