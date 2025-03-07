from setuptools import setup, find_packages
import tomli

with open('pyproject.toml', 'rb') as f:
    version = tomli.load(f)['project']['version']

setup(
    name='whatifact',
    version=version,
    packages=find_packages() + ['whatifact.resources'],
    test_suite='tests',
    url='http://github.com/urigott/whatifact',
    author_email='urigott@gmail.com',
    license='MIT',
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=1.1.0",
        "shiny>=1.0.0",
    ]

)