from setuptools import setup, find_packages

setup(
    name='whatifact',
    version='0.1.0',
    packages=find_packages(),
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