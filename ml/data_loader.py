"""
ml/data_loader.py

Loads the 13 NHANES 2017-2018 XPT files from data/raw/, merges them
on SEQN via inner join, prints shape and missing-value counts,
and saves the merged result to data/interim/merged.csv.
"""

import os
import pyreadstat
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ------------------------------------------python ml/data_loader.py---------------------------------

# Root of the project (parent of ml/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
INTERIM_DIR = os.path.join(PROJECT_ROOT, "data", "interim")

# The 13 NHANES 2017-2018 files to load and merge
NHANES_FILES = [
    "DEMO_J",
    "VITAEC_J",
    "VIC_J",
    "VID_J",
    "FERTIN_J",
    "FETIB_J",
    "TFR_J",
    "FOLATE_J",
    "FOLFMS_J",
    "PBCD_J",
    "HSCRP_J",
    "CBC_J",
    "BIOPRO_J",
]

MERGE_KEY = "SEQN"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_xpt(filename):
    """Load a single NHANES XPT file from data/raw/ as a DataFrame."""
    path = os.path.join(RAW_DIR, f"{filename}.XPT")

    if not os.path.exists(path):
        # Try lowercase extension as fallback
        alt = os.path.join(RAW_DIR, f"{filename}.xpt")
        if os.path.exists(alt):
            path = alt
        else:
            raise FileNotFoundError(
                f"Could not find '{filename}.XPT' (or .xpt) in {RAW_DIR}. "
                f"Download it from the CDC NHANES 2017-2018 website."
            )

    df, _ = pyreadstat.read_xport(path)
    return df


def load_and_merge():
    """Load all 13 files, inner-join them on SEQN, and return the merged DataFrame."""
    dfs = []

    for filename in NHANES_FILES:
        print(f"Loading {filename} ...")
        df = load_xpt(filename)
        print(f"  {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        dfs.append(df)

    # Sequential inner join on SEQN
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=MERGE_KEY, how="inner")

    return merged


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_missing(df):
    """Print missing-value counts for every column, sorted highest to lowest."""
    missing = df.isna().sum().sort_values(ascending=False)

    print("\n" + "=" * 60)
    print("MISSING VALUE COUNTS (sorted highest to lowest)")
    print("=" * 60)

    for column, count in missing.items():
        print(f"{column:<15} {count}")

    return missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Ensure output directory exists
    os.makedirs(INTERIM_DIR, exist_ok=True)

    # Load and merge
    merged = load_and_merge()

    # Report shape
    print("\n" + "=" * 60)
    print("MERGED DATAFRAME SHAPE")
    print("=" * 60)
    print(f"Rows:    {merged.shape[0]}")
    print(f"Columns: {merged.shape[1]}")

    # Report missing values
    report_missing(merged)

    # Save merged result
    output_path = os.path.join(INTERIM_DIR, "merged.csv")
    merged.to_csv(output_path, index=False)
    print(f"\nSaved merged dataframe to: {output_path}")

    return merged


if __name__ == "__main__":
    main()