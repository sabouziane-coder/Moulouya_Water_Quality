#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
FEATURES=["T°C","pH","EC","DO","BOD5","PO4","NH4","SO4","NO3"]
def main(input_csv, output_fig):
    df = pd.read_csv(input_csv)
    # create label
    if "Station" in df.columns and "Campaign" in df.columns:
        df["_x"] = df["Station"].astype(str) + " | " + df["Campaign"].astype(str)
    else:
        df["_x"] = df.index.astype(str)
    x = range(len(df))
    plt.figure(figsize=(14,7))
    for f in FEATURES:
        col = f if f in df.columns else f+"_scaled"
        plt.plot(x, df[col].values, marker='o', label=f)
    plt.xticks(x, df["_x"].tolist(), rotation=90, fontsize=8)
    plt.legend(bbox_to_anchor=(1.02,1), loc='upper left')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved figure to", output_fig)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args=parser.parse_args()
    main(args.input, args.output)

