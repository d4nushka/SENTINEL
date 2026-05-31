"""
cross_domain.py
===============
Evaluates SENTINEL performance across each product category separately.
This is a key table in the research paper showing generalizability.

Run:
    python src/cross_domain.py
"""
import pickle
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef
from scipy.sparse import hstack, csr_matrix

FEATURE_COLS = ['word_count','char_count','avg_word_len','exclamation_cnt',
                'question_cnt','caps_ratio','type_token_ratio','repetition_score',
                'sentiment_score','vagueness_score','burstiness','perplexity_proxy',
                'avg_sent_len','punct_ratio','rating']

def evaluate_category(name, y_true, y_pred, y_prob=None):
    if len(y_true) < 10:
        return None
    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc   = matthews_corrcoef(y_true, y_pred)
    auc   = roc_auc_score(y_true, y_prob) if y_prob is not None and len(set(y_true)) > 1 else 0
    return {
        "category": name,
        "n_samples": len(y_true),
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "kappa": round(kappa, 4),
        "mcc": round(mcc, 4),
    }

def run_cross_domain():
    print("[cross_domain] Loading models and data...")
    baseline  = pickle.load(open("outputs/model_baseline.pkl",  "rb"))
    gbm       = pickle.load(open("outputs/model_gbm.pkl",       "rb"))
    sentinel  = pickle.load(open("outputs/model_sentinel.pkl",  "rb"))
    scaler    = pickle.load(open("outputs/scaler.pkl",          "rb"))
    tfidf_vec = pickle.load(open("outputs/tfidf_vec.pkl",       "rb"))

    df = pd.read_csv("data/reviews.csv")

    # Use only MR2 data for cross-domain (has proper categories)
    df_mr2 = df[df['category'] != 'Mixed'].copy()
    print(f"[cross_domain] Using {len(df_mr2)} reviews across {df_mr2.category.nunique()} categories")

    # Re-extract features for this subset
    import sys
    sys.path.insert(0, '.')
    from src.features import extract_features
    df_mr2 = extract_features(df_mr2)

    categories = df_mr2['category'].unique()
    results_baseline = []
    results_sentinel = []

    for cat in sorted(categories):
        subset = df_mr2[df_mr2['category'] == cat]
        if len(subset) < 20:
            continue

        X_text = subset['text'].astype(str)
        X_feat = subset[FEATURE_COLS].fillna(0)
        y_true = subset['label'].values

        # Baseline
        y_pred_bl = baseline.predict(X_text)
        y_prob_bl = baseline.predict_proba(X_text)[:, 1]
        r = evaluate_category(cat, y_true, y_pred_bl, y_prob_bl)
        if r: results_baseline.append(r)

        # SENTINEL Hybrid
        X_scaled  = scaler.transform(X_feat)
        X_tfidf   = tfidf_vec.transform(X_text)
        X_hybrid  = hstack([X_tfidf, csr_matrix(X_scaled)])
        y_pred_s  = sentinel.predict(X_hybrid)
        y_prob_s  = sentinel.predict_proba(X_hybrid)[:, 1]
        r = evaluate_category(cat, y_true, y_pred_s, y_prob_s)
        if r: results_sentinel.append(r)

    # Results tables
    df_bl  = pd.DataFrame(results_baseline)
    df_sen = pd.DataFrame(results_sentinel)

    print("\n── Baseline (TF-IDF + LR) Cross-Domain Results ──")
    print(df_bl[['category','n_samples','accuracy','f1','kappa','mcc']].to_string(index=False))

    print("\n── SENTINEL Hybrid Cross-Domain Results ──────────")
    print(df_sen[['category','n_samples','accuracy','f1','kappa','mcc']].to_string(index=False))

    print("\n── Average Performance ───────────────────────────")
    print(f"  Baseline  — Avg F1: {df_bl['f1'].mean():.4f} | Avg Kappa: {df_bl['kappa'].mean():.4f}")
    print(f"  SENTINEL  — Avg F1: {df_sen['f1'].mean():.4f} | Avg Kappa: {df_sen['kappa'].mean():.4f}")

    # Save results
    Path("outputs").mkdir(exist_ok=True)
    df_bl.to_csv("outputs/cross_domain_baseline.csv", index=False)
    df_sen.to_csv("outputs/cross_domain_sentinel.csv", index=False)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    categories_sorted = df_bl.sort_values('f1')['category'].tolist()
    bl_f1  = df_bl.set_index('category').loc[categories_sorted, 'f1']
    sen_f1 = df_sen.set_index('category').loc[categories_sorted, 'f1']

    x = np.arange(len(categories_sorted))
    width = 0.35

    axes[0].barh(x - width/2, bl_f1,  width, label='Baseline',       color='#5B8DB8')
    axes[0].barh(x + width/2, sen_f1, width, label='SENTINEL Hybrid', color='#E8663D')
    axes[0].set_yticks(x)
    axes[0].set_yticklabels([c.replace(' and ', ' & ') for c in categories_sorted], fontsize=9)
    axes[0].set_xlabel('F1 Score')
    axes[0].set_title('F1 Score by Product Category')
    axes[0].legend()
    axes[0].set_xlim(0.7, 1.0)
    axes[0].axvline(0.93, color='gray', linestyle='--', alpha=0.5, label='Overall avg')

    bl_kappa  = df_bl.set_index('category').loc[categories_sorted, 'kappa']
    sen_kappa = df_sen.set_index('category').loc[categories_sorted, 'kappa']
    axes[1].barh(x - width/2, bl_kappa,  width, label='Baseline',       color='#5B8DB8')
    axes[1].barh(x + width/2, sen_kappa, width, label='SENTINEL Hybrid', color='#E8663D')
    axes[1].set_yticks(x)
    axes[1].set_yticklabels([c.replace(' and ', ' & ') for c in categories_sorted], fontsize=9)
    axes[1].set_xlabel("Cohen's Kappa")
    axes[1].set_title("Cohen's Kappa by Product Category")
    axes[1].legend()
    axes[1].set_xlim(0.4, 1.0)

    plt.suptitle("SENTINEL Cross-Domain Evaluation", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/cross_domain_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[cross_domain] Chart saved -> outputs/cross_domain_results.png")
    print("Done! Results saved to outputs/cross_domain_*.csv")

if __name__ == "__main__":
    run_cross_domain()