import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/raw")

for item in DATA_DIR.rglob("*"):
    if item.is_file():
        print(item.relative_to(DATA_DIR))