import os, io, base64, re, joblib, pandas as pd, numpy as np
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tensorflow.keras.models import load_model
from feature_engineering import norm, parse_list, avg_hit_ratio, is_big_studio, is_sequel, overview_sentiment, compute_star_power, BoosterWrapper
from utils_constants import MAX_RELEASE_YEAR
import shap
from feature_labels import NICE_FEAT


# Load artifacts
cls_model = load_model("cls_model.keras")
reg_model = joblib.load("reg_model.pkl")
qt_scaler = joblib.load("qt_scaler.pkl")
core_cols = joblib.load("feature_columns_core.pkl")
reg_cols = joblib.load("feature_columns_reg.pkl")
mlb_genre = joblib.load("mlb_genre.pkl")
mlb_lang = joblib.load("mlb_lang.pkl")
tfidf = joblib.load("overview_tfidf.pkl")
entity_year = pd.read_parquet("entity_star_power.parquet")
hit_ratio = joblib.load("director_hit_ratio_map.pkl")
calibrator = joblib.load("prob_calibrator.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse input
        rd = pd.to_datetime(request.form.get("release_date",""), errors="coerce")
        ry = int(rd.year) if pd.notnull(rd) else 0
        ry  = min(ry, MAX_RELEASE_YEAR) 
        rm = int(rd.month) if pd.notnull(rd) else 0
        budget = float(request.form.get("budget", 0) or 0)
        runtime = float(request.form.get("runtime", 0) or 0)
        overview = request.form.get("overview", "")

        cast = parse_list(request.form.get("cast", ""))
        direc = parse_list(request.form.get("director", ""))
        writers = parse_list(request.form.get("writers", ""))
        producers = parse_list(request.form.get("producers", ""))
        companies = parse_list(request.form.get("production_companies", ""))
        countries = parse_list(request.form.get("production_countries", ""))
        genres = parse_list(request.form.get("genres", ""))
        langs = parse_list(request.form.get("spoken_languages", ""))
        title = norm(request.form.get("title", ""))

        base = {
            "runtime": runtime,
            "log_budget": np.log1p(budget),
            "overview_sentiment": overview_sentiment(overview),
            "release_year": ry,
            "release_month": rm,
            "is_holiday_release": int(rm in [11,12]),
            "big_studio": is_big_studio(companies),
            "is_sequel": is_sequel(title),
            "cast_power": compute_star_power(entity_year, cast, ry, top_k=3),
            "director_power": compute_star_power(entity_year, direc, ry),
            "writers_power": compute_star_power(entity_year, writers, ry),
            "producers_power": compute_star_power(entity_year, producers, ry),
            "production_companies_power": compute_star_power(entity_year, companies, ry),
            "production_countries_power": compute_star_power(entity_year, countries, ry),
            "director_hit_ratio": (float(np.mean([hit_ratio.get(d, 0.0) for d in direc])) if direc else 0.0),
        }
        df0 = pd.DataFrame([base])

        g_df = pd.DataFrame(mlb_genre.transform([genres]), columns=[f"genre_{g}" for g in mlb_genre.classes_])
        l_df = pd.DataFrame(mlb_lang.transform([langs]), columns=[f"lang_{l}" for l in mlb_lang.classes_])
        t_df = pd.DataFrame(tfidf.transform([overview]).toarray(), columns=[f"tfidf_{i}" for i in range(tfidf.transform([overview]).shape[1])])

        # Downscale TF-IDF features (IMPORTANT)
        t_df *= 0.003      
        df0["overview_sentiment"] = np.clip(df0["overview_sentiment"], -0.5, 0.5)

        # After creating g_df and l_df
        assert all(col in core_cols for col in g_df.columns), "Genre mismatch!"
        assert all(col in core_cols for col in l_df.columns), "Language mismatch!"

        raw = pd.concat([df0, g_df, l_df, t_df], axis=1)
        for c in core_cols:
            if c not in raw.columns:
                raw[c] = 0

        scaled = qt_scaler.transform(raw[qt_scaler.feature_names_in_])
        fin = pd.concat([pd.DataFrame(scaled, columns=qt_scaler.feature_names_in_), raw.drop(columns=qt_scaler.feature_names_in_)], axis=1)

        # Save user input for debugging
        os.makedirs("debug_inputs", exist_ok=True)
        user_input_file = f"debug_inputs/user_input_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv"
        fin[core_cols].to_csv(user_input_file, index=False)
        print(f"✅ User input saved to {user_input_file}")

        raw_p = float(cls_model.predict(fin[core_cols])[0, 0])
        ann_p = float(calibrator.predict([raw_p])[0])
        label = "🔥 Blockbuster" if ann_p >= 0.46 else "💤 Flop"

        X_reg = fin[core_cols].copy()
        X_reg["success_prob"] = ann_p
        rev = np.expm1(reg_model.predict(X_reg[reg_cols])[0])
        low, high = int(rev*0.85), int(rev*1.15)

        # ─── Per-movie SHAP explanation ──────────────────────────────────
        explainer = shap.TreeExplainer(reg_model.booster)
        shap_vals = explainer.shap_values(X_reg[reg_cols])
        row_vals  = shap_vals[0]                         # first (only) row
        top_idx   = np.argsort(np.abs(row_vals))[-5:]    # top-5 absolute impacts
        top5 = (pd.DataFrame({
                    "f": np.array(reg_cols)[top_idx],
                    "i": row_vals[top_idx]
                }).sort_values("i")) 
        top5["label"] = top5["f"].map(NICE_FEAT).fillna(top5["f"])
        #  ─── Cool bar-chart  ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(5, 4))

        # Modern lollipop chart: green for positive, red for negative
        colors = ["#4caf50" if v > 0 else "#f44336" for v in top5["i"]]

        for idx, (feat, val) in enumerate(zip(top5["label"], top5["i"])):
            ax.plot([0, val], [idx, idx], color=colors[idx], lw=2)   # stem line
            ax.scatter(val, idx, color=colors[idx], s=60, edgecolor="black", zorder=3)  # dot

        # Set feature names as y-ticks
        ax.set_yticks(range(len(top5)))
        ax.set_yticklabels(top5["label"], fontsize=10)
        ax.set_xlabel("Impact on Revenue Prediction", fontsize=12)
        #ax.set_title("Top 5 Influencers", fontsize=14, weight="bold", pad=15)

        # Clean grid and frame
        ax.grid(True, axis="x", linestyle="--", alpha=0.7)
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="both", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        img = base64.b64encode(buf.getvalue()).decode()


        return render_template("form.html", prediction={
            "label": label,
            "performance": f"{ann_p*100:.2f}%",
            "success_probability": f"{ann_p*100:.2f}%",
            "estimated_revenue": f"${low:,} – ${high:,}",
            "importance_chart": f"data:image/png;base64,{img}"
        })

    except Exception as e:
        return render_template("form.html", error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
