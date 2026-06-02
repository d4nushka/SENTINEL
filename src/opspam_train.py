"""
opspam_train.py
===============
Trains and evaluates SENTINEL on the OpSpam dataset.
Standard protocol: 80% train, 20% test — same as all published papers.

Run:
    python src/opspam_train.py
"""
import pickle
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score,
                             cohen_kappa_score, matthews_corrcoef,
                             classification_report)
from scipy.sparse import hstack, csr_matrix
import sys
sys.path.insert(0, '.')
from src.features import extract_features

FEATURE_COLS = ['word_count','char_count','avg_word_len','exclamation_cnt',
                'question_cnt','caps_ratio','type_token_ratio','repetition_score',
                'sentiment_score','vagueness_score','burstiness','perplexity_proxy',
                'avg_sent_len','punct_ratio','rating']

def evaluate(name, y_true, y_pred, y_prob=None):
    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc   = matthews_corrcoef(y_true, y_pred)
    auc   = roc_auc_score(y_true, y_prob) if y_prob is not None else 0
    print(f"\n── {name} ──────────────────────────────────")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  ROC-AUC     : {auc:.4f}")
    print(f"  Cohen Kappa : {kappa:.4f}")
    print(f"  MCC         : {mcc:.4f}")
    print(classification_report(y_true, y_pred, target_names=['Real','Fake']))
    return {"model": name, "accuracy": round(acc,4), "f1": round(f1,4),
            "auc": round(auc,4), "kappa": round(kappa,4), "mcc": round(mcc,4)}

def run():
    print("[opspam_train] Loading OpSpam splits...")
    train = pd.read_csv("data/opspam_train.csv")
    test  = pd.read_csv("data/opspam_test.csv")

    for df in [train, test]:
        df['label']  = df['deceptive'].apply(lambda x: 1 if x == 'deceptive' else 0)
        df['text']   = df['text'].astype(str).str.strip()
        df['rating'] = 4.0

    print(f"[opspam_train] Train: {len(train)} | Test: {len(test)}")

    train = extract_features(train)
    test  = extract_features(test)

    X_train_text = train['text']
    X_test_text  = test['text']
    X_train_feat = train[FEATURE_COLS].fillna(0)
    X_test_feat  = test[FEATURE_COLS].fillna(0)
    y_train = train['label']
    y_test  = test['label']

    results = []
    Path("outputs").mkdir(exist_ok=True)

    # Model 1: Baseline
    print("\n[opspam_train] Training Baseline...")
    baseline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)),
        ('clf',   LogisticRegression(max_iter=1000, C=1.0, random_state=42))
    ])
    baseline.fit(X_train_text, y_train)
    y_pred = baseline.predict(X_test_text)
    y_prob = baseline.predict_proba(X_test_text)[:,1]
    results.append(evaluate("Baseline TF-IDF + LR", y_test, y_pred, y_prob))

    # Model 2: Linguistic + GBM
    print("\n[opspam_train] Training Linguistic + GBM...")
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train_feat)
    X_te_s = scaler.transform(X_test_feat)
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                      learning_rate=0.1, random_state=42)
    gbm.fit(X_tr_s, y_train)
    y_pred = gbm.predict(X_te_s)
    y_prob = gbm.predict_proba(X_te_s)[:,1]
    results.append(evaluate("Linguistic + GBM", y_test, y_pred, y_prob))

    # Model 3: Hybrid
    print("\n[opspam_train] Training SENTINEL Hybrid...")
    tfidf_v = TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)
    X_tr_tfidf = tfidf_v.fit_transform(X_train_text)
    X_te_tfidf = tfidf_v.transform(X_test_text)
    X_tr_h = hstack([X_tr_tfidf, csr_matrix(X_tr_s)])
    X_te_h = hstack([X_te_tfidf, csr_matrix(X_te_s)])
    hybrid = LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver='saga')
    hybrid.fit(X_tr_h, y_train)
    y_pred = hybrid.predict(X_te_h)
    y_prob = hybrid.predict_proba(X_te_h)[:,1]
    results.append(evaluate("SENTINEL Hybrid", y_test, y_pred, y_prob))

    # Summary
    print("\n── OpSpam Results (80/20 split) ─────────────────")
    df_r = pd.DataFrame(results)
    print(df_r[['model','accuracy','f1','auc','kappa','mcc']].to_string(index=False))

    print("\n── vs Published SOTA on OpSpam ──────────────────")
    sota = [
        ("Ott et al. 2011 (SVM)",      0.896, 0.895),
        ("Li et al. 2014 (Neural)",    0.886, 0.884),
        ("Rashkin et al. 2017 (LSTM)", 0.912, 0.910),
        ("Best BERT-based",            0.946, 0.944),
    ]
    for name, acc, f1 in sota:
        print(f"  {name:<35} Acc={acc} F1={f1}")

    best = max(results, key=lambda x: x['f1'])
    print(f"\n  Ours best: {best['model']:<27} Acc={best['accuracy']} F1={best['f1']}")

    if best['f1'] >= 0.944:
        print("  🏆 BEATS published SOTA!")
    elif best['f1'] >= 0.90:
        print("  ✓ Competitive with top published results!")
    else:
        print(f"  Gap to SOTA: {round((0.944-best['f1'])*100,1)}%")

    df_r.to_csv("outputs/opspam_train_results.csv", index=False)
    json.dump(results, open("outputs/opspam_train_results.json","w"), indent=2)
    print("\nDone! Results saved to outputs/opspam_train_results.csv")

if __name__ == "__main__":
    run()