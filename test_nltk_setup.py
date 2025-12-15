"""
Simple script to test if NLTK is properly set up for the book-to-audiobook script.
Run this to verify NLTK installation and punkt tokenizer availability.
"""

import sys

print("Testing NLTK setup for book-to-audiobook script...")
print("=" * 60)

# Test 1: Check if NLTK is installed
try:
    import nltk
    print("[OK] NLTK is installed")
except ImportError:
    print("[ERROR] NLTK is not installed")
    print("  Install with: pip install nltk")
    sys.exit(1)

# Test 2: Check if punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
    print("[OK] NLTK punkt tokenizer is available")
except LookupError:
    print("[WARNING] NLTK punkt tokenizer is not found")
    print("  Attempting to download...")
    try:
        nltk.download('punkt', quiet=False)
        # Verify it was downloaded
        try:
            nltk.data.find('tokenizers/punkt')
            print("[OK] NLTK punkt tokenizer downloaded successfully")
        except LookupError:
            print("[ERROR] Download may have failed. Please try manually:")
            print("  python -c \"import nltk; nltk.download('punkt')\"")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Could not download punkt tokenizer: {e}")
        print("  Try manually: python -c \"import nltk; nltk.download('punkt')\"")
        sys.exit(1)

# Test 3: Test sentence tokenization
try:
    from nltk.tokenize import sent_tokenize
    test_text = "This is a test. It has multiple sentences! Does it work? Yes, it does."
    sentences = sent_tokenize(test_text)
    if len(sentences) == 4:
        print("[OK] Sentence tokenization is working correctly")
        print(f"  Test text split into {len(sentences)} sentences as expected")
    else:
        print(f"[WARNING] Sentence tokenization returned {len(sentences)} sentences (expected 4)")
        print(f"  Sentences: {sentences}")
except Exception as e:
    print(f"[ERROR] Sentence tokenization failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("[SUCCESS] NLTK is properly set up and ready to use!")
print("  The book-to-audiobook script will use NLTK for sentence splitting.")

