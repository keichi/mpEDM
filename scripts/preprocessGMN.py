#!/usr/bin/env python3

import numpy as np
import pandas as pd
import h5py

import sys, argparse
from pathlib import Path

parser = argparse.ArgumentParser( description = 'Convert HDF5 to CSV for GMN' )

parser.add_argument('-i', '--inputFile',
                    dest    = 'inputFile',
                    type    = str,
                    action  = 'store',
                    help    = 'Input file of CCM result from mpEDM')

parser.add_argument('-d', '--dataset',
                    dest    = 'dataset',
                    type    = str,
                    action  = 'store',
                    help    = 'Input file of original dataset')

args = parser.parse_args()

result_file = h5py.File(args.inputFile, 'r')
result_df = pd.DataFrame(result_file["corrcoef"])

if Path(args.dataset).suffix == ".h5":
    dataset_file = h5py.File(args.dataset, 'r')
    dataset_names = pd.DataFrame(dataset_file["names"]).stack().str.decode('utf-8')
    dataset_values = pd.DataFrame(dataset_file["values"])

    dataset_values.columns = dataset_names
    dataset_values.index += 1
    dataset_values.to_csv(Path(args.dataset).with_suffix(".csv"), index = True)
    
    result_df.columns = dataset_names
    result_df.index = dataset_names
    result_df.to_csv(Path(args.inputFile).with_suffix(".csv"), index = True)
else: 
    print("The dataset is not in the HDF5(.h5) format.")