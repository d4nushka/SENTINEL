import pickle
import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

FEATURE_COLS = ['word_count','char_count','avg_word_len','exclamation_cnt',
                'question_cnt','caps_ratio','type_token_ratio','repetition_score',
                'sentiment_score','vagueness_score','burstiness','perplexity_proxy',
                'avg_sent_len','punct_ratio','rating']

def run_shap():
    print("[explain] Loading model and data...")
    df     = pd.read_csv("data/test_features.csv")
    gbm    = pickle.load(open("outputs/model_gbm.pkl", "rb"))
    scaler = pickle.load(open("outputs/scaler.pkl",    "rb"))

    X = pd.DataFrame(scaler.transform(df[FEATURE_COLS].fillna(0)), columns=FEATURE_COLS)

    print("[explain] Computing SHAP values (1-2 mins)...")
    explainer   = shap.TreeExplainer(gbm)
    shap_values = explainer.shap_values(X)

    Path("outputs").mkdir(exist_ok=True)

    # Handle both old and new shap output formats
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif len(np.array(shap_values).shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X, show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[explain] Saved -> outputs/shap_summary.png")

    # Bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[explain] Saved -> outputs/shap_bar.png")

    # Feature importance table
    mean_shap = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature':    FEATURE_COLS,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)

    print("\n── SHAP Feature Importance ──────────────────────")
    print(importance_df.to_string(index=False))
    importance_df.to_csv("outputs/shap_importance.csv", index=False)
    print("\nDone! Run next: streamlit run app.py")

if __name__ == "__main__":
    run_shap()