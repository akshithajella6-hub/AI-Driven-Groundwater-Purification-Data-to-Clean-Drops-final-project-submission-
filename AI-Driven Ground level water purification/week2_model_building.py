import pandas as pd

df = pd.read_csv("synthetic_groundwater.csv")

thresholds = {
    "pH": (6.5, 8.5),
    "TDS": (None, 500),
    "turbidity": (None, 5),
    "nitrate": (None, 50),
    "arsenic": (None, 0.01),
    "lead": (None, 0.01),
    "fluoride": (None, 1.5),
    "hardness": (None, 300),
    "temperature": (None, 35),
    "EC": (None, 1500),
    "coliform": (0, 0)
}

def within_threshold(value, low, high):
    if pd.isna(value):
        return False
    if low is not None and value < low:
        return False
    if high is not None and value > high:
        return False
    return True

df["safe_water"] = df.apply(
    lambda row: int(all(within_threshold(row[col], *thresholds[col]) for col in thresholds if col in df.columns)),
    axis=1
)

df.to_csv("processed_with_target.csv", index=False)
print(" Target column added. File saved as processed_with_target.csv")
