#!/usr/bin/env python3
"""Transcribe handwritten TEXT LINE images with a released HTR-JAND model (line-level recognizer).

Thin wrapper around `htrjand.predict_cli` (also installed as the `htrjand-predict` command);
`python predict.py --help` lists every option.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(HERE, "htrjand")):
    sys.path.insert(0, HERE)

from htrjand.predict_cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
