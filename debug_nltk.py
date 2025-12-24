
import nltk
from nltk import word_tokenize, pos_tag
import sys

print(f"NLTK Version: {nltk.__version__}")
print(f"NLTK Path: {nltk.data.path}")

print("\nChecking resources...")
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
    print("Found averaged_perceptron_tagger")
except LookupError as e:
    print(f"Missing averaged_perceptron_tagger: {e}")

try:
    nltk.data.find('tokenizers/punkt')
    print("Found punkt")
except LookupError as e:
    print(f"Missing punkt: {e}")

try:
    nltk.data.find('tokenizers/punkt_tab')
    print("Found punkt_tab")
except LookupError as e:
    print(f"Missing punkt_tab: {e}")

print("\nTesting functionality...")
try:
    tokens = word_tokenize("Test sentence.")
    print(f"Tokenization successful: {tokens}")
    try:
        tags = pos_tag(tokens)
        print(f"POS Tagging successful: {tags}")
    except Exception as e:
        print(f"POS Tagging failed: {e}")
except Exception as e:
    print(f"Tokenization failed: {e}")
