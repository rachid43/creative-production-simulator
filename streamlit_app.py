"""Entry point for Streamlit Community Cloud (its default main-file name). Runs the simplified app in app.py."""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py"), run_name="__main__")
