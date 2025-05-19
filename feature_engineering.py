import re
import numpy as np
import pandas as pd
from unidecode import unidecode
from textblob import TextBlob
from utils_constants import BIG_STUDIOS, SEQUEL_TOKENS
import xgboost as xgb

class BoosterWrapper:
    def __init__(self, booster, cols):
        self.booster = booster
        self.cols = cols
    def predict(self, X):
        return self.booster.predict(xgb.DMatrix(X[self.cols]))
    
class KerasProbWrapper:
    """Expose .predict_proba for CalibratedClassifierCV."""
    def __init__(self, keras_model):
        self.model = keras_model
    def fit(self, X, y):
        # No fitting – the ANN is already trained
        return self
    def predict_proba(self, X):
        p = self.model.predict(X, verbose=0).ravel()
        return np.column_stack([1 - p, p])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


TOKEN = re.compile(r"[A-Za-z0-9]+")

# Normalize text (lowercase + remove accents)
def norm(x: str) -> str:
    return unidecode(x).lower().strip()

# Parse comma-separated lists
def parse_list(s: str) -> list[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [norm(x) for x in s.split(",") if x.strip()]

# Big studio flag
def is_big_studio(companies: list[str]) -> int:
    if not companies:
        return 0
    tokens = {tok for c in companies for tok in TOKEN.findall(c)}
    return int(any(bs in tokens for bs in BIG_STUDIOS))

# Sequel flag
def is_sequel(title: str) -> int:
    return int(any(tok in norm(title) for tok in SEQUEL_TOKENS))

# Sentiment score
def overview_sentiment(text: str) -> float:
    return TextBlob(text).sentiment.polarity if text else 0.0

# Star-power computation
def compute_star_power(entity_year: pd.DataFrame, entities: list[str], year: int, top_k: int = None) -> float:
    vals = (entity_year.loc[(entity_year["entity"].isin(entities)) & (entity_year["year"] <= year)]
                        .groupby("entity")["star_power"].last().fillna(0).to_numpy())
    if top_k:
        vals = np.sort(vals)[-top_k:][::-1]
        if len(vals) == top_k:
            vals *= np.array([0.6, 0.3, 0.1])
    return float(vals.sum())

def avg_hit_ratio(dir_list):
    if not dir_list:
        return 0.0
    vals = [succ_map.get(d, 0.0) for d in dir_list]
    return float(np.mean(vals))
