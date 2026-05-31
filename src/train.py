import pandas as pd
import numpy as np
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, f1_score, accuracy_score,
                             roc_auc_score, matthews_corrcoef, cohen_kappa_score)
from scipy.sparse import hstack, csr_matrix

FEATURE_COLS = ['word_count','char_count','avg_word_len','exclamation_cnt',
                'question_cnt','caps_ratio','type_token_ratio','repetition_score',
                'sentiment_score','vagueness_score','burstiness','perplexity_proxy',
                'avg_sent_len','punct_ratio','rating']

def evaluate(name, y_true, y_pred, y_prob=None):
    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred)
    mcc   = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    auc   = roc_auc_score(y_true, y_prob) if y_prob is not None else 0
    print(f"\n── {name} ──────────────────────────────")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  F1 Score     : {f1:.4f}")
    print(f"  ROC-AUC      : {auc:.4f}")
    print(f"  Cohen Kappa  : {kappa:.4f}")
    print(f"  MCC          : {mcc:.4f}")
    print(classification_report(y_true, y_pred, target_names=['Real','Fake']))
    return {"model": name, "accuracy": round(acc,4), "f1": round(f1,4),
            "auc": round(auc,4), "kappa": round(kappa,4), "mcc": round(mcc,4)}

def train_all():
    print("[train] Loading data...")
    train = pd.read_csv("data/train_features.csv")
    test  = pd.read_csv("data/test_features.csv")

    X_train_text = train['text'].astype(str)
    X_test_text  = test['text'].astype(str)
    X_train_feat = train[FEATURE_COLS].fillna(0)
    X_test_feat  = test[FEATURE_COLS].fillna(0)
    y_train = train['label']
    y_test  = test['label']

    print(f"[train] Train: {len(train)} | Test: {len(test)}")

    results = []
    Path("outputs").mkdir(exist_ok=True)

    # ── Model 1: Baseline TF-IDF + Logistic Regression ──────────────────────
    print("\n[train] Model 1: TF-IDF + Logistic Regression (baseline)...")
    baseline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)),
        ('clf',   LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])
    baseline.fit(X_train_text, y_train)
    y_pred = baseline.predict(X_test_text)
    y_prob = baseline.predict_proba(X_test_text)[:,1]
    results.append(evaluate("Baseline: TF-IDF + LR", y_test, y_pred, y_prob))
    pickle.dump(baseline, open("outputs/model_baseline.pkl", "wb"))

    # ── Model 2: Linguistic Features + Gradient Boosting ────────────────────
    print("\n[train] Model 2: Linguistic Features + XGBoost...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled  = scaler.transform(X_test_feat)
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                      learning_rate=0.1, random_state=42)
    gbm.fit(X_train_scaled, y_train)
    y_pred = gbm.predict(X_test_scaled)
    y_prob = gbm.predict_proba(X_test_scaled)[:,1]
    results.append(evaluate("Linguistic + GBM", y_test, y_pred, y_prob))
    pickle.dump(gbm,    open("outputs/model_gbm.pkl",    "wb"))
    pickle.dump(scaler, open("outputs/scaler.pkl",       "wb"))

    # ── Model 3: SENTINEL Hybrid (TF-IDF + Linguistic Features) ─────────────
    print("\n[train] Model 3: SENTINEL Hybrid...")
    tfidf_vec = TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)
    X_train_tfidf = tfidf_vec.fit_transform(X_train_text)
    X_test_tfidf  = tfidf_vec.transform(X_test_text)
    X_train_hybrid = hstack([X_train_tfidf, csr_matrix(X_train_scaled)])
    X_test_hybrid  = hstack([X_test_tfidf,  csr_matrix(X_test_scaled)])
    from sklearn.linear_model import LogisticRegression as LR
    sentinel = LR(max_iter=1000, C=1.0, random_state=42, solver='saga')
    sentinel.fit(X_train_hybrid, y_train)
    y_pred = sentinel.predict(X_test_hybrid)
    y_prob = sentinel.predict_proba(X_test_hybrid)[:,1]
    results.append(evaluate("SENTINEL Hybrid", y_test, y_pred, y_prob))
    pickle.dump(sentinel,  open("outputs/model_sentinel.pkl", "wb"))
    pickle.dump(tfidf_vec, open("outputs/tfidf_vec.pkl",      "wb"))

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n── Final Model Comparison ───────────────────────")
    results_df = pd.DataFrame(results)
    print(results_df[['model','accuracy','f1','auc','kappa','mcc']].to_string(index=False))
    results_df.to_csv("outputs/model_results.csv", index=False)
    json.dump(results, open("outputs/model_results.json","w"), indent=2)

    best = max(results, key=lambda x: x['f1'])
    print(f"\n✓ Best model: {best['model']}")
    print(f"  F1={best['f1']} | Kappa={best['kappa']} | MCC={best['mcc']}")
    print("\nDone! Run next: python src/explain.py")

if __name__ == "__main__":
    train_all()