#!/usr/bin/env python3

import numpy as np
import pandas as pd
import h5py

import sys
from pathlib import Path

path = Path(sys.argv[1])
df = pd.read_csv(path, dtype=np.float32)

with h5py.File(path.with_suffix(".h5"), "w") as f:
    f.create_dataset(name="names", data=df.columns, dtype=h5py.string_dtype())
    f.create_dataset(name="values", data=df)
