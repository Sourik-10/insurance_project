import pandas as pd

RAW_PATH = "data/insurance.csv"
CLEAN_PATH = "data/insurance_clean.csv"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    print(f"Raw shape: {df.shape}")
    print(f"Missing values:\n{df.isna().sum()}")

    n_dupes = df.duplicated().sum()
    print(f"Duplicate rows: {n_dupes}")
    df = df.drop_duplicates().reset_index(drop=True)

    # Standardize categorical text (defensive - this dataset is already clean,
    # but don't assume that for any CSV you load)
    df["sex"] = df["sex"].str.lower().str.strip()
    df["smoker"] = df["smoker"].str.lower().str.strip()
    df["region"] = df["region"].str.lower().str.strip()

    # Sanity checks
    assert df["age"].between(0, 120).all()
    assert df["bmi"].gt(0).all()
    assert df["charges"].gt(0).all()
    assert df["smoker"].isin(["yes", "no"]).all()

    print(f"Final shape: {df.shape}")
    print(f"Charges range: {df['charges'].min():.2f} - {df['charges'].max():.2f}")

    return df


if __name__ == "__main__":
    df = load_and_clean(RAW_PATH)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned data to {CLEAN_PATH}")
