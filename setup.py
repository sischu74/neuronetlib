#!/usr/bin/env python3
from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="neuronetlib",
    version="0.0.1",
    author="Sascha Kehrli",
    author_email="skehrli@vis.ethz.ch",
    description="Neural network library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sischu74/neuronetlib",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
