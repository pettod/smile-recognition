"""setup.py

Used with `pip install .`, installs the requirements and source file directory
"""
from setuptools import setup, find_packages

setup(
    name='Smile Detection',
    install_requires=list(open('requirements.txt').readlines()),
    author='Jesper Granat, Peter Todorov',
    version='1.0',
    packages=find_packages()
)
