#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the function directly from the test file
import importlib.util
spec = importlib.util.spec_from_file_location("test_analyzer", "tests/test_analyzer.py")
test_analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_analyzer)

# Also import the actual function to see what it returns
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyzer import get_disfluency_rate

# Test the specific case
text2 = "um uh like you know hello world"
print(f"Testing text: {repr(text2)}")

rate2 = get_disfluency_rate(text2)
print(f"get_disfluency_rate returned: {rate2}")
print(f"Expected: 0.4")
print(f"Difference: {abs(rate2 - 0.4)}")

# Let's also manually calculate what we think it should be
import re
fillers = len(re.findall(r'\b(um|uh|like|you know)\b', text2, re.IGNORECASE))
words = len(text2.split())
manual_rate = fillers / max(words, 1)
print(f"Manual calculation: {fillers} fillers / {words} words = {manual_rate}")

try:
    test_analyzer.test_get_disfluency_rate_basic()
    print("Test passed!")
except Exception as e:
    print(f"Test failed with: {e}")