#!/usr/bin/env python3
"""
preprocess.py
- Loads Excel file
- Converts features to float
- Splits into train/test
- Fits MinMax scaler on training set and saves processed CSV
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

FEATURES = ["T°C","pH","EC","DO","BOD5","PO4","NH4","SO4","NO3"]

def main(input_excel, output_csv, random_seed=42, test_size=0.2):
    df = pd.read_excel(input_excel)
    # minimal column detection: adjust names if needed
    # convert features robustly
    for f in FEATURES:
        df[f] = df[f].astype(str).str.replace(",", ".").str.strip()
        df[f] = pd.to_numeric(df[f], errors='coerce').astype(float)
    # split
    train_idx, test_idx = train_test_split(df.index, test_size=test_size, random_state=random_seed)
    scaler = MinMaxScaler()
    scaler.fit(df.loc[train_idx, FEATURES].values)
    scaled = scaler.transform(df[FEATURES].values)
    for i,f in enumerate(FEATURES):
        df[f+"_scaled"] = scaled[:,i]
    # save scaler and processed csv
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    joblib.dump(scaler, os.path.join(os.path.dirname(output_csv), "minmax_scaler.pkl"))
    print("Saved processed dataset and scaler.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", required=True)
    parser.add_argument("--output", dest="output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)

