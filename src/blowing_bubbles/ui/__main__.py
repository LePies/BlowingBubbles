"""Launch the Streamlit UI via ``python -m blowing_bubbles.ui``."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app = Path(__file__).with_name("app.py")
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)]))
