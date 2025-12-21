#!/usr/bin/env python3
import sys
import os

# Add src directory to python path to allow relative imports
# This allows running the script directly from the project root
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

try:
    from coquitts.main import main
except ImportError as e:
    print(f"Error importing coquitts package: {e}")
    print("Ensure you are running this script from the project root directory")
    print("and that the 'src' directory structure is correct.")
    sys.exit(1)

if __name__ == "__main__":
    main()
