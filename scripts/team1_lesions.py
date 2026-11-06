"""Team 1 steps 0 and 1: build the lesion table and the two reference centres files.

Usage: PYTHONPATH=src python scripts/team1_lesions.py RAW TEAM1_DIR
Writes TEAM1_DIR/lesions.csv and
  centres/bl.csv          step 0: FU window at the BL lesion's mm position (same patient coordinates)
  centres/propagated.csv  step 1: cog_propagated from the dataset, else the BL position (the bar to beat)
Score them with scripts/team1_eval.py. Only reads image headers, so a few minutes on one CPU.
"""
import argparse
from pathlib import Path

import numpy as np

from lesionloc.team1 import cols, lesion_table, write_centres


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = lesion_table(a.raw)
    df.to_csv(out / "lesions.csv", index=False, float_format="%.3f")
    print("lesions per split", df.groupby("split").size().to_dict(),
          "patients", df.groupby("split")["patient"].nunique().to_dict())
    print("has_propagated", int(df["has_propagated"].sum()), "of", len(df))

    bl = df[cols("bl")].to_numpy()
    prop = np.where(df[["has_propagated"]].to_numpy(), df[cols("prop")].to_numpy(), bl)
    write_centres(df, bl, out / "centres" / "bl.csv")
    write_centres(df, prop, out / "centres" / "propagated.csv")


if __name__ == "__main__":
    main()
