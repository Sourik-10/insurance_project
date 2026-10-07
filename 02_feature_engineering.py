import pandas as pd

CLEAN_PATH = "data/insurance_clean.csv"
FEATURED_PATH = "data/insurance_featured.csv"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standard WHO BMI categories (not arbitrary - these are the real clinical cutoffs)
    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, 100],
        labels=["underweight", "normal", "overweight", "obese"],
    )

    # The interaction that actually matters in this data: smoker AND obese.
    # (See 03_eda.py - non-smokers' charges barely vary by BMI category at all;
    # the cost explosion only happens for smokers crossing the obesity line.)
    df["smoker_obese"] = (
        (df["smoker"] == "yes") & (df["bmi"] >= 30)
    ).astype(int)

    # Simple age bands - charges rise with age but not sharply (corr=0.30),
    # bands mainly help tree-based models split cleanly
    df["age_group"] = pd.cut(
        df["age"], bins=[17, 25, 35, 45, 55, 65],
        labels=["18-25", "26-35", "36-45", "46-55", "56-64"],
    )

    df["has_children"] = (df["children"] > 0).astype(int)

    return df


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH)
    df = add_features(df)
    df.to_csv(FEATURED_PATH, index=False)
    print(f"Saved featured data to {FEATURED_PATH}")
    print(df[["bmi_category", "smoker_obese", "age_group", "has_children"]].describe(include="all"))
