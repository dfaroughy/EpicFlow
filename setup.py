import os
import re
import subprocess
import setuptools

def load_requirements():
    try:
        with open("requirements.txt") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("WARNING: requirements.txt not found")
        return []
try:
    with open("README.md", "r") as f:
        long_description = f.read()
except:
    long_description = "# Epic Flow "

setuptools.setup(
    name="EpicFlow",
    version="1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="git@github.com:dfaroughy/EpicFlow.git",
    packages=setuptools.find_packages("source"),
    package_dir={"": "source"},
    python_requires=">=3.7",
    install_requires=load_requirements(),
    include_package_data=True
)
