"""
ablation.py
===========
Ablation study — shows contribution of each feature group to SENTINEL performance.
Removes one feature group at a time and measures performance drop.
This is Table 4 in the research paper.

Run:
    python src/ablation.py
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

ALL_FEATURES = ['word_count','char_count','avg_word_len','exclamation_cnt',
                'question_cnt','caps_ratio','type_token_ratio','repetition_score',
                'sentiment_score','vagueness_score','burstiness','perplexity_proxy',
                'avg_sent_len','punct_ratio','rating']

# Feature groups for ablation
FEATURE_GROUPS = {
    "Length features":      ['word_count', 'char_count', 'avg_word_len', 'avg_sent_len'],
    "Punctuation features": ['exclamation_cnt', 'question_cnt', 'caps_ratio', 'punct_ratio'],
    "Lexical features":     ['type_token_ratio', 'repetition_score'],
    "Sentiment features":   ['sentiment_score', 'vagueness_score'],
    "Chaos features":       ['burstiness', 'perplexity_proxy'],
    "Rating":               ['rating'],
}

def train_and_eval(X_train, y_train, X_test, y_test, features):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train[features].fillna(0))
    X_te = scaler.transform(X_test[features].fillna(0))
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    return {
        "f1":    round(f1_score(y_test, y_pred), 4),
        "kappa": round(cohen_kappa_score(y_test, y_pred), 4),
        "mcc":   round(matthews_corrcoef(y_test, y_pred), 4),
        "auc":   round(roc_auc_score(y_test, y_prob), 4),
    }

def run_ablation():
    print("[ablation] Loading data...")
    train = pd.read_csv("data/train_features.csv")
    test  = pd.read_csv("data/test_features.csv")
    y_train = train['label']
    y_test  = test['label']

    results = []

    # Full model baseline
    print("[ablation] Training full model...")
    full = train_and_eval(train, y_train, test, y_test, ALL_FEATURES)
    full['experiment'] = "Full model (all features)"
    full['removed']    = "None"
    full['n_features'] = len(ALL_FEATURES)
    results.append(full)
    print(f"  Full model — F1={full['f1']} | Kappa={full['kappa']}")

    # Remove one group at a time
    for group_name, group_features in FEATURE_GROUPS.items():
        remaining = [f for f in ALL_FEATURES if f not in group_features]
        print(f"[ablation] Without {group_name}...")
        r = train_and_eval(train, y_train, test, y_test, remaining)
        r['experiment'] = f"Without {group_name}"
        r['removed']    = group_name
        r['n_features'] = len(remaining)
        r['f1_drop']    = round(full['f1'] - r['f1'], 4)
        r['kappa_drop'] = round(full['kappa'] - r['kappa'], 4)
        results.append(r)
        print(f"  F1={r['f1']} | Kappa={r['kappa']} | F1 drop={r['f1_drop']}")

    # Individual features only
    print("[ablation] Testing each feature group alone...")
    for group_name, group_features in FEATURE_GROUPS.items():
        r = train_and_eval(train, y_train, test, y_test, group_features)
        r['experiment'] = f"Only {group_name}"
        r['removed']    = f"All except {group_name}"
        r['n_features'] = len(group_features)
        r['f1_drop']    = round(full['f1'] - r['f1'], 4)
        r['kappa_drop'] = round(full['kappa'] - r['kappa'], 4)
        results.append(r)
        print(f"  Only {group_name}: F1={r['f1']} | Kappa={r['kappa']}")

    df = pd.DataFrame(results)

    print("\n── Ablation Study Results ───────────────────────")
    ablation_rows = df[df['experiment'].str.startswith('Without')]
    print(ablation_rows[['experiment','n_features','f1','kappa','f1_drop','kappa_drop']].to_string(index=False))

    print("\n── Individual Feature Groups ────────────────────")
    solo_rows = df[df['experiment'].str.startswith('Only')]
    print(solo_rows[['experiment','n_features','f1','kappa']].to_string(index=False))

    # Save
    Path("outputs").mkdir(exist_ok=True)
    df.to_csv("outputs/ablation_results.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ablation_plot = df[df['experiment'].str.startswith('Without')].copy()
    ablation_plot = ablation_plot.sort_values('f1_drop', ascending=True)
    colors = ['#e74c3c' if x > 0.01 else '#f39c12' if x > 0.005 else '#2ecc71'
              for x in ablation_plot['f1_drop']]
    bars = ax.barh(ablation_plot['removed'], ablation_plot['f1_drop'], color=colors)
    ax.set_xlabel("F1 Score Drop (higher = more important feature group)")
    ax.set_title("Ablation Study — Feature Group Importance\n(SENTINEL Linguistic Model)")
    ax.axvline(0, color='black', linewidth=0.8)
    for bar, val in zip(bars, ablation_plot['f1_drop']):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/ablation_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[ablation] Plot saved -> outputs/ablation_plot.png")
    print("Done! Results saved to outputs/ablation_results.csv")

if __name__ == "__main__":
    run_ablation()