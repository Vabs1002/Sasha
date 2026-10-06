#!/usr/bin/env python3
import re

text2 = "um uh like you know hello world"
print(f"Text: {repr(text2)}")

# Find all matches
matches = re.findall(r'\b(um|uh|like|you know)\b', text2, re.IGNORECASE)
print(f"Matches: {matches}")
print(f"Number of matches: {len(matches)}")

# Find match positions
for match in re.finditer(r'\b(um|uh|like|you know)\b', text2, re.IGNORECASE):
    print(f"Match: {repr(match.group())} at position {match.start()}-{match.end()}")

# Let's also see what the individual word matches would be
um_matches = re.findall(r'\bum\b', text2, re.IGNORECASE)
uh_matches = re.findall(r'\buh\b', text2, re.IGNORECASE)
like_matches = re.findall(r'\blike\b', text2, re.IGNORECASE)
you_know_matches = re.findall(r'\byou know\b', text2, re.IGNORECASE)

print(f"'um' matches: {um_matches} (count: {len(um_matches)})")
print(f"'uh' matches: {uh_matches} (count: {len(uh_matches)})")
print(f"'like' matches: {like_matches} (count: {len(like_matches)})")
print(f"'you know' matches: {you_know_matches} (count: {len(you_know_matches)})")

# Total filler word count if we count actual words
total_filler_words = len(um_matches) + len(uh_matches) + len(like_matches) + (len(you_know_matches) * 2)
print(f"Total filler words (counting 'you know' as 2): {total_filler_words}")

# Total words by split
words = text2.split()
print(f"Total words by split: {len(words)} {words}")

# Rate if we count actual filler words
if len(words) > 0:
    rate_actual_filler_words = total_filler_words / len(words)
    print(f"Rate (actual filler words): {total_filler_words}/{len(words)} = {rate_actual_filler_words}")

# Rate if we count matcher groups
if len(words) > 0:
    rate_match_groups = len(matches) / len(words)
    print(f"Rate (match groups): {len(matches)}/{len(words)} = {rate_match_groups}")