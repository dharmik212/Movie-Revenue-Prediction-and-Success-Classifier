import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("TMDB_movie_dataset_Full.csv", low_memory=False)

cols = [
    "id", "title", "release_date", "vote_average", "vote_count", "runtime", 
    "popularity", "spoken_languages", "genres", "overview",
    "production_companies", "production_countries",
    "cast", "director", "writers", "producers",
    "imdb_rating", "imdb_votes", "budget", "revenue"
]
df = df[cols]

df = df.dropna(subset=cols)
df = df[df.runtime != 0]
df = df[(df.budget >= 1000) & (df.revenue > 0)]

# Binary label
df["label"] = (df["revenue"] > 1.5 * df["budget"]).astype(int)

# Log-scaled numerical columns
df["log_budget"] = np.log1p(df["budget"])
df["log_revenue"] = np.log1p(df["revenue"])

df.to_csv("TMDB_filtered_labeled.csv", index=False)

# Class distribution plot
counts = df['label'].value_counts().sort_index()
counts.plot(kind='bar')
plt.title("Class Distribution of Label")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()
