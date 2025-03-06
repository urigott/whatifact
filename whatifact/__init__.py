from .whatifact import whatifact
import tomli

with open('pyproject.toml', 'rb') as f:
    version = tomli.load(f)['project']['version']

__all__ = ["whatifact"]
__author__ = "Uri Gottlieb <urigott@gmail.com>"
__version__ = version
