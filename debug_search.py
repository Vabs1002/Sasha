#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interviewer_agent import search_resume_tool

resume_text = """
John Doe
Software Engineer

Experience:
- Built web applications using Python and React
- Improved system performance by 40% through optimization
- Worked with AWS cloud services

Skills: Python, JavaScript, React, AWS, Docker

Projects:
- Created a real-time chat application with WebSockets
- Developed a recommendation engine using machine learning
"""

print("Testing search_resume_tool with 'Python'...")
results = search_resume_tool("Python", resume_text)
print(f"Results: {results}")
print(f"Length: {len(results)}")

print("\nTesting search_resume_tool with 'web applications'...")
results2 = search_resume_tool("web applications", resume_text)
print(f"Results: {results2}")
print(f"Length: {len(results2)}")