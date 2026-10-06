#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Read the test file directly
with open('tests/test_analyzer.py', 'r') as f:
    lines = f.readlines()

# Find the test
for i, line in enumerate(lines):
    if 'text2 = "um uh like you know hello world"' in line:
        print(f"Line {i+1}: {repr(line)}")
        # Also get the surrounding lines
        for j in range(max(0, i-2), min(len(lines), i+3)):
            marker = ">>> " if j == i else "    "
            print(f"{marker}{j+1}: {repr(lines[j])}")
        break