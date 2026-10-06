"""
Setup script for Sasha - Adaptive AI Interviewer
"""
from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename) as f:
        lines = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)
        return lines

requirements = parse_requirements('requirements.txt')

setup(
    name="sasha-ai-interviewer",
    version="1.0.0",
    author="Vaibhav Mittal",
    author_email="vaibhav.mittal@example.com",
    description="A resume‑driven, agentic AI interviewer that behaves like a senior technical recruiter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sasha-ai-interviewer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "sasha-interviewer=main:main",
        ],
    },
    include_package_data=True,
)