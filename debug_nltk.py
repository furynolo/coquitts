
import sys
import nltk

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

text = "Korin approached."
text2 = "Herac and Estes approached,"
tokens = nltk.word_tokenize(text)
tags = nltk.pos_tag(tokens)
print(f"'{text}': {tags}")

tokens2 = nltk.word_tokenize(text2)
tags2 = nltk.pos_tag(tokens2)
print(f"'{text2}': {tags2}")
