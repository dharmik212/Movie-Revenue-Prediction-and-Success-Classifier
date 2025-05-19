import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load feature data
df = pd.read_csv("movie_features.csv")

# Select only original numerical + encoded date/sentiment features
cols = [
    "cast_power", "director_power", "writers_power", "producers_power",
    "production_companies_power", "production_countries_power",
    "overview_sentiment", "runtime", "log_budget",
    "release_year", "release_month", "is_holiday_release"
]

# Histograms
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 12))
axes = axes.flatten()
for ax, col in zip(axes, cols):
    ax.hist(df[col].dropna(), bins=30)
    ax.set_title(f"Histogram of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
fig.tight_layout()
plt.show()

# Violin plots by label
fig2, axes2 = plt.subplots(nrows=3, ncols=4, figsize=(20, 12))
axes2 = axes2.flatten()
for ax, col in zip(axes2, cols):
    sns.violinplot(x="label", y=col, data=df, ax=ax)
    ax.set_title(f"{col} by Label")
fig2.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(df[cols + ["log_revenue"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix (Numerical Features Only)")
plt.show()
