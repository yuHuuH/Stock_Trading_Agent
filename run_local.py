from pathlib import Path
import subprocess
import sys

base = Path(__file__).resolve().parent
app = base / "app.py"

subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)])
