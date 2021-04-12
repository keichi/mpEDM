#!/usr/bin/env python3

import numpy as np
import pandas as pd
import h5py

import sys
from pathlib import Path

path = Path(sys.argv[1])
df = h5py.File(path, 'r')

for key in list(df):
    data = pd.DataFrame(list(df[key]))
    data.to_csv(path.with_suffix("." + key +".csv"), index = False)