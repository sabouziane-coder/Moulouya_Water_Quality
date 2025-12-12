#!/usr/bin/env python3
import argparse
import pandas as pd
import joblib
import os

def compute_weights(rw):
    s = sum(rw)
    return [r/s for r in rw]

def main(input_csv, norms_excel, output_csv):
    df = pd.read_csv(input_csv)
    norms = pd.read_excel(norms_excel, sheet_name="Normes")  # adjust sheet name
    # norms expected to have columns: Parameter, Si, Rw
    norms = norms.set_index("Parameter")
    Rw = norms["Rw"].to_dict()
    Si = norms["Si"].to_dict()
    # compute normalized weights
    rws = [Rw[p] for p in df.columns if p in Rw]  # careful, better mapping
    total_rw = sum(Rw.values())
    W = {p: Rw[p]/total_rw for p in Rw}
    # compute PI
    def calc_pi(row):
        total=0.0
        for p in W:
            Ci = row.get(p)
            Si_v = Si[p]
            total += W[p] * (Ci/Si_v)
        return total
    df["PI_calc"] = df.apply(calc_pi, axis=1)
    df.to_csv(output_csv, index=False)
    print("Saved PI-calculated dataset to", output_csv)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--norms", required=True)
    parser.add_argument("--output", required=True)
    args=parser.parse_args()
    main(args.input, args.norms, args.output)

