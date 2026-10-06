#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzer import get_disfluency_rate

# Text with fillers
text2 = "um uh like you know hello world"
print(f"Text: '{text2}'")
print(f"Length: {len(text2)}")
print(f"Repr: {repr(text2)}")

rate2 = get_disfluency_rate(text2)
print(f"Rate: {rate2}")
print(f"Expected: 0.4")
print(f"Match: {rate2 == 0.4}")

# Let's manually calculate what it should be
import re
fillers = len(re.findall(r'\b(um|uh|like|you know)\b', text2, re.IGNORECASE))
words = len(text2.split())
print(f"Fillers: {fillers}")
print(f"Words: {words}")
print(f"Calculated rate: {fillers}/{words} = {fillers/words}")