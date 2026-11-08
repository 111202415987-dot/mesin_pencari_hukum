"""
app_rag.py
==========
Pintu masuk yang disebut di notebook (`streamlit run app_rag.py`).
Seluruh antarmuka ada di app.py; file ini hanya menjalankannya ulang di setiap rerun Streamlit.
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
runpy.run_path(str(HERE / "app.py"), run_name="__main__")