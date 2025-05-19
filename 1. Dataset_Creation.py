import os
import pandas as pd

MAIN_FILE      = "TMDB_movie_dataset_v11.csv"
SECONDARY_CSV  = "TMDB_all_movies_1M.csv"
OUTPUT_CSV     = "TMDB_movie_dataset_Full.csv"

EXTRA_COLS = [
    "cast",
    "director",
    "director_of_photography",
    "writers",
    "producers",
    "music_composer",
    "imdb_rating",
    "imdb_votes",
]

def read_table(path, usecols=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx"):
        return pd.read_excel(path, usecols=usecols)
    else:
        try:
            return pd.read_csv(path, usecols=usecols, encoding="utf-8")
        except UnicodeDecodeError:
            print(f"⚠️  UnicodeDecodeError reading {path}, retrying with latin-1")
            return pd.read_csv(path, usecols=usecols, encoding="latin-1")

def main():
    df_main = read_table(MAIN_FILE)
    df_sec = read_table(SECONDARY_CSV, usecols=["id"] + EXTRA_COLS)
    df_merged = df_main.merge(df_sec, on="id", how="left")
    df_merged.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Wrote enriched file to {OUTPUT_CSV} ({len(df_merged)} rows)")

if __name__ == "__main__":
    main()
