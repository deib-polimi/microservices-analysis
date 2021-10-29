import nltk
import sys
import string
from collections import Counter

def keywords(data):
        data = data.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        data = data.translate(str.maketrans(string.digits, ' '*len(string.digits)))
        data = data.lower()
        words = [w for w in nltk.word_tokenize(data) if len(w) > 1]
        counter = Counter(words)
        most_commons = [x[0] for x in counter.most_common(5)]
        print(most_commons)

filename = 'ex1.txt'
with open(filename, 'r') as f:
    keywords(f.read())
    