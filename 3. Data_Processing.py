import os, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from feature_engineering import norm, parse_list, avg_hit_ratio, is_big_studio, is_sequel, overview_sentiment
from utils_constants import BIG_STUDIOS, SEQUEL_TOKENS, MAX_RELEASE_YEAR
from textblob import TextBlob
from unidecode import unidecode
import re

RAW_CSV = "TMDB_filtered_labeled.csv"
PARQUET_PATH = "entity_star_power.parquet"

df = pd.read_csv(RAW_CSV, low_memory=False)

# Lowercasing + normalize
for col in ["cast", "director", "writers", "producers", "production_companies", "production_countries", "title", "genres", "spoken_languages"]:
    df[col] = df[col].apply(norm)

roles = ["cast","director","writers","producers","production_companies","production_countries"]
for r in roles:
    df[f"{r}_list"] = df[r].apply(parse_list)

df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["release_year"] = (
    pd.to_datetime(df["release_date"], errors="coerce")
      .dt.year.fillna(0)
      .clip(upper=MAX_RELEASE_YEAR)    
)
df["release_month"] = df["release_date"].dt.month.fillna(0)
df["is_holiday_release"] = df["release_month"].isin([11,12]).astype(int)

# Rebuild star-power if needed
if (not os.path.exists(PARQUET_PATH) or os.path.getmtime(RAW_CSV) > os.path.getmtime(PARQUET_PATH)):
    print("▶ rebuilding star-power cache")
    frames=[]
    for r in roles:
        tmp=(df[["id","revenue","budget","vote_average","vote_count",
                 "popularity","imdb_rating","imdb_votes",f"{r}_list"]]
             .explode(f"{r}_list").rename(columns={f"{r}_list":"entity"}))
        tmp["entity"] = tmp["entity"].apply(norm)
        tmp["role"] = r
        tmp["profit"] = tmp["revenue"]-tmp["budget"]
        tmp["tmdb_weighted"] = tmp["vote_average"]*tmp["vote_count"]
        tmp["imdb_weighted"] = tmp["imdb_rating"]*tmp["imdb_votes"]
        frames.append(tmp)

    clong = pd.concat(frames, ignore_index=True)
    clong = clong.merge(df[["id","release_date"]], on="id")
    clong["release_year"] = pd.to_datetime(clong["release_date"], errors="coerce").dt.year

    def build_star_table(cl, entity):
        cl = cl.sort_values("release_year")
        rows = []
        agg = {"profit":0, "tmdb_weighted":0, "imdb_weighted":0, "popularity":0}
        current_year = None
        for _, row in cl.iterrows():
            if row.release_year != current_year:
                current_year = row.release_year
                rows.append({"entity": entity, "year": current_year, **agg})
            agg["profit"] += row.profit
            agg["tmdb_weighted"] += row.tmdb_weighted
            agg["imdb_weighted"] += row.imdb_weighted
            agg["popularity"] += row.popularity
        return pd.DataFrame(rows)

    ent_yearly = clong.groupby("entity", group_keys=False).apply(lambda g: build_star_table(g, g.name)).reset_index(drop=True)
    ent_yearly["star_power"] = ent_yearly["profit"] + ent_yearly["tmdb_weighted"] + ent_yearly["imdb_weighted"] + ent_yearly["popularity"]
    ent_yearly.to_parquet(PARQUET_PATH)
else:
    ent_yearly = pd.read_parquet(PARQUET_PATH)
    print("✓ loaded star-power cache")

# Compute powers
from feature_engineering import compute_star_power
df["cast_power"] = [compute_star_power(ent_yearly, ents, yr, top_k=3) for ents, yr in zip(df["cast_list"], df["release_year"])]
df["director_power"] = [compute_star_power(ent_yearly, ents, yr) for ents, yr in zip(df["director_list"], df["release_year"])]
df["writers_power"] = [compute_star_power(ent_yearly, ents, yr) for ents, yr in zip(df["writers_list"], df["release_year"])]
df["producers_power"] = [compute_star_power(ent_yearly, ents, yr) for ents, yr in zip(df["producers_list"], df["release_year"])]
df["production_companies_power"] = [compute_star_power(ent_yearly, ents, yr) for ents, yr in zip(df["production_companies_list"], df["release_year"])]
df["production_countries_power"] = [compute_star_power(ent_yearly, ents, yr) for ents, yr in zip(df["production_countries_list"], df["release_year"])]

# Flags
df["big_studio"] = df["production_companies_list"].apply(is_big_studio)
df["is_sequel"] = df["title"].apply(is_sequel)

# Director hit ratio
dir_long = (
    df[['director_list', 'label']]
    .explode('director_list')
    .rename(columns={'director_list': 'director_ind'})
)
succ_map = dir_long.groupby('director_ind')['label'].mean().to_dict()
joblib.dump(succ_map, "director_hit_ratio_map.pkl")

df['director_hit_ratio'] = df['director_list'].apply(
    lambda L: float(np.mean([succ_map.get(d, 0.0) for d in L])) if L else 0.0
)

# Multi-hot encodings
mlb_gen = MultiLabelBinarizer()
genre_df = pd.DataFrame(mlb_gen.fit_transform(df["genres"].apply(parse_list)),
                        columns=[f"genre_{g}" for g in mlb_gen.classes_])
joblib.dump(mlb_gen,"mlb_genre.pkl")

mlb_lang = MultiLabelBinarizer()
language_df = pd.DataFrame(mlb_lang.fit_transform(df["spoken_languages"].apply(parse_list)),
                            columns=[f"lang_{l}" for l in mlb_lang.classes_])
joblib.dump(mlb_lang,"mlb_lang.pkl")

tfidf = TfidfVectorizer(max_features=50)
tfidf_mat = tfidf.fit_transform(df["overview"].fillna("")).toarray()
tfidf_df = pd.DataFrame(tfidf_mat, columns=[f"tfidf_{i}" for i in range(tfidf_mat.shape[1])])
joblib.dump(tfidf,"overview_tfidf.pkl")

df["overview_sentiment"] = df["overview"].fillna("").apply(overview_sentiment)
df["overview_sentiment"] = df["overview_sentiment"].clip(-0.5, 0.5)

# Scaling numeric features
num_cols = ["cast_power","director_power","writers_power","producers_power",
            "production_companies_power","production_countries_power",
            "overview_sentiment","release_year","release_month",
            "is_holiday_release","big_studio","is_sequel","director_hit_ratio",
            "runtime","log_budget"]

train_idx,_ = train_test_split(df.index,test_size=0.3,random_state=42,stratify=df["label"])
qt = QuantileTransformer(output_distribution="normal",random_state=42)
qt.fit(df.loc[train_idx,num_cols])
df[num_cols] = qt.transform(df[num_cols])
joblib.dump(qt,"qt_scaler.pkl")

# Save
final = pd.concat([df[["id"]+num_cols+["log_revenue","label"]],genre_df,language_df,tfidf_df],axis=1)
core_cols = final.drop(columns=["id","log_revenue","label"]).columns.tolist()
final.to_csv("movie_features.csv",index=False)
joblib.dump(core_cols,"feature_columns_core.pkl")
print("✓ movie_features.csv & artefacts saved")
