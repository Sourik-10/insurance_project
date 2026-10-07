import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
FEATURED_PATH = "data/insurance_featured.csv"
OUT_DIR = "outputs"


def main():
    df = pd.read_csv(FEATURED_PATH)

    # ---- 1. Smoker vs non-smoker charges ----
    smoker_means = df.groupby("smoker")["charges"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="smoker", y="charges", ax=ax)
    ax.set_title("Charges: smokers vs non-smokers")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/charges_by_smoker.png")
    plt.close(fig)
    ratio = smoker_means["yes"] / smoker_means["no"]
    print(f"[Smoker] Smokers pay ${smoker_means['yes']:,.0f} on average vs "
          f"${smoker_means['no']:,.0f} for non-smokers — a {ratio:.1f}x "
          f"difference. This is by far the single strongest driver of cost "
          f"in this dataset.")

    # ---- 2. BMI x smoker interaction ----
    interaction = df.groupby(["smoker", "bmi_category"], observed=True)["charges"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 4))
    interaction.T.plot(kind="bar", ax=ax)
    ax.set_ylabel("Average charges ($)")
    ax.set_title("BMI category vs charges, split by smoking status")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/bmi_smoker_interaction.png")
    plt.close(fig)
    non_smoker_spread = interaction.loc["no"].max() - interaction.loc["no"].min()
    smoker_spread = interaction.loc["yes"].max() - interaction.loc["yes"].min()
    print(f"[BMI x Smoker] Among non-smokers, average charges barely move "
          f"across BMI categories (spread of only ${non_smoker_spread:,.0f}). "
          f"Among smokers, the spread is ${smoker_spread:,.0f} — BMI only "
          f"matters once someone already smokes. This is the real interaction "
          f"worth encoding as a feature, not BMI or smoking status alone.")

    # ---- 3. Age vs charges ----
    corr = df["age"].corr(df["charges"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x="age", y="charges", hue="smoker", alpha=0.6, ax=ax)
    ax.set_title("Age vs charges, colored by smoking status")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/age_vs_charges.png")
    plt.close(fig)
    print(f"[Age] Age correlates with charges at only r={corr:.2f} — a weak "
          f"positive relationship. Age matters much less than smoking status; "
          f"an older non-smoker often costs less than a younger smoker.")

    # ---- 4. Children ----
    child_means = df.groupby("children")["charges"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    child_means.plot(kind="bar", ax=ax, color="#55A868")
    ax.set_ylabel("Average charges ($)")
    ax.set_title("Number of children vs average charges")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/children_vs_charges.png")
    plt.close(fig)
    print(f"[Children] No clean monotonic trend — charges don't rise steadily "
          f"with number of dependents (e.g. 5 children averages "
          f"${child_means.get(5, 0):,.0f}, lower than 2-3 children). Sample "
          f"sizes get small at the high end, so treat this feature as weak "
          f"evidence, not a strong driver.")

    # ---- 5. Region ----
    region_means = df.groupby("region")["charges"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    region_means.plot(kind="bar", ax=ax, color="#DD8452")
    ax.set_ylabel("Average charges ($)")
    ax.set_title("Average charges by region")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/region_vs_charges.png")
    plt.close(fig)
    print(f"[Region] '{region_means.index[0]}' has the highest average charges "
          f"(${region_means.iloc[0]:,.0f}), but the spread across all four "
          f"regions is only ${region_means.max() - region_means.min():,.0f} — "
          f"region is a minor factor compared to smoking status.")

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
