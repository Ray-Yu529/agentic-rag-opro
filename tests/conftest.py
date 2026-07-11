"""tests/conftest.py — 讓 tests/ 能 import 專案根目錄的模組。
全部測試離線 (不打 API、不下載資料集)，CI 不需要 NVIDIA_API_KEY。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
