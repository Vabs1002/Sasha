#!/usr/bin/env python3
import sys
import os
import re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

text2 = "um uh like you know hello world"
print(f"Text: '{text2}'")

# Test the regex
fillers = len(re.findall(r'\b(um|uh|like|you know)\b', text2, re.IGNORECASE))
print(f"Fillers found: {fillers}")

words = len(text2.split())
print(f"Words (split): {words}")
print(f"Split words: {text2.split()}")

rate = fillers / max(words, 1)
print(f"Rate: {rate}")

# Let's also test what the test expects
expected_rate = 0.4
print(f"Expected rate: {expected_rate}")
print(f"Expected fillers for {expected_rate * words} = {expected_rate * words}")