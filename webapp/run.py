#!/usr/bin/env python3
"""Start Kronos Market Foresight (Vansh Bhasin / bhasinvansh05)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import app  # noqa: E402


def main():
    port = int(os.environ.get("PORT", 7070))
    print(f"Kronos webapp → http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
