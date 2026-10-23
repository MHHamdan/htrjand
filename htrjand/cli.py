"""`htrjand-predict` console entry point (same as predict.py)."""
import sys

from .predict_cli import main as _main


def main():
    sys.exit(_main())
