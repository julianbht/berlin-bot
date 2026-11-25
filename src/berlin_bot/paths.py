from pathlib import Path

# The app and the download script use paths relative to the project root ("datasets/...").
# Works locally and on Streamlit Community Cloud, as long as the package is installed editable.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
