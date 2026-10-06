#!/usr/bin/env python3
text2 = "um uh like you know hello world"
print(f"Bytes: {list(text2.encode())}")
print(f"Length: {len(text2)}")
print(f"Repr: {repr(text2)}")

# Check for trailing spaces
if text2.endswith(' '):
    print("Ends with space")
else:
    print("Does not end with space")

if text2.startswith(' '):
    print("Starts with space")
else:
    print("Does not start with space")

# Count spaces
space_count = text2.count(' ')
print(f"Space count: {space_count}")

# What if we count words differently?
# What if we split on spaces and then filter empty strings?
words_split = text2.split()
words_split_filter = [w for w in text2.split() if w]
print(f"split(): {words_split} (count: {len(words_split)})")
print(f"split() filtered: {words_split_filter} (count: {len(words_split_filter)})")

# What about splitting on whitespace (including tabs, newlines)?
import re
words_re_split = re.split(r'\s+', text2)
words_re_split_filter = [w for w in re.split(r'\s+', text2) if w]
print(f"regex split: {words_re_split} (count: {len(words_re_split)})")
print(f"regex split filtered: {words_re_split_filter} (count: {len(words_re_split_filter)})")