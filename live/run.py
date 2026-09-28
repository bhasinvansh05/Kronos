#!/usr/bin/env python3
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from app import main

if __name__ == "__main__":
    main()
