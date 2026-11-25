"""Entry point for `uv run berlin-bot`: starts the Streamlit app."""

import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli

APP_PATH = Path(__file__).resolve().parent / "app.py"


def main() -> None:
    # Streamlit only runs apps through its own CLI: hand over to it, extra arguments included
    sys.argv = ["streamlit", "run", str(APP_PATH), *sys.argv[1:]]
    sys.exit(streamlit_cli.main())
