"""Data Processor for AI-Driven Groundwater Purification
    - Loads a groundwater CSV dataset
    - Cleans missing values
    - Normalizes numerical features
    - Exports processed data to a new CSV
    - Generates a quick summary report
"""

import argparse
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Define the features we expect in groundwater quality datasets
FEATURES = [ "pH", "TDS", "turbidity", "nitrate", "arsenic", "lead", "fluoride", "hardness", "temperature", "EC", "coliform"]

def generate_report(df, report_path="report.txt"):
    """Generate a simple text report of dataset statistics."""
    with open(report_path, "w", encoding="utf-8") as f:  #  force UTF-8
        f.write(" Groundwater Data Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n\n")
        f.write("Statistics:\n")
        f.write(str(df.describe()) + "\n")
    print(f"[INFO] Saved report → {report_path}")

def process_data(input_path: str, output_path: str):
    """Load, clean, normalize, and save the groundwater dataset."""
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"[ERROR] File not found: {input_path}")
        return
    # Load dataset
    df = pd.read_csv(input_path)
    print(f"[INFO] Loaded dataset with shape {df.shape}")
    # Check columns
    missing_cols = set(FEATURES) - set(df.columns)
    if missing_cols:
        print(f"[WARNING] Missing expected columns: {missing_cols}")
    available_features = [f for f in FEATURES if f in df.columns]
    # Fill missing values (numeric median)
    df[available_features] = df[available_features].fillna(df[available_features].median())
    # Normalize numeric features
    scaler = StandardScaler()
    df[available_features] = scaler.fit_transform(df[available_features])
    # Save processed dataset
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved processed dataset → {output_path}")
    # Generate report
    generate_report(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Groundwater Data Processor")
    parser.add_argument("--input", default="raw_groundwater.csv", help="Path to raw CSV dataset")
    parser.add_argument("--output", default="processed_groundwater.csv", help="Path to save processed CSV")
    args = parser.parse_args()

    process_data(args.input, args.output)



