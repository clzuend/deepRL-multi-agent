from setuptools import setup, Command, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(packages=find_packages(),
      install_requires = required,
)