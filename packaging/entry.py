"""Entry script for the PyInstaller build.

`src/lode/__main__.py` cannot be used directly: PyInstaller compiles the analysis
target as a top-level script, so its relative `from .cli import main` fails at runtime.
"""

from lode.cli import main

main()
