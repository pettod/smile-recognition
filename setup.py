from setuptools import setup, find_packages

setup(
    name='Smile Detection',
    install_requires=list(open('requirements.txt').readlines()),
    author='Jesper Granat, Peter Todorov',
    version='0.1',
    packages=find_packages()
)