"""
opspam_eval.py
==============
Evaluates SENTINEL on the OpSpam (Deceptive Opinion Spam) dataset.
This is the standard benchmark used by every fake review paper.

Key: Our model was trained on Amazon product reviews (MR2).
OpSpam is hotel reviews from TripAdvisor — completely different domain.
Good performance here = true cross-domain generalization.

Run:
    python src/opspam_eval.py
"""
import pickle
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (f1_score, accuracy_score, roc_auc_score,
                             cohen_kappa_score, matthews_corrcoef,
                             classification_report, confusion_matrix)
from scipy.sparse import hstack, csr_matrix
import sys
sys.path.insert(0, '.')
from src.features import extract_features

FEATURE_COLS = ['word_count','char_count','avg_word_len','exclamation_cnt',
                'question_cnt','caps_ratio','type_token_ratio','repetition_score',
                'sentiment_score','vagueness_score','burstiness','perplexity_proxy',
                'avg_sent_len','punct_ratio','rating']

def run_opspam_eval():
    print("[opspam] Loading OpSpam dataset...")
    df = pd.read_csv("data/deceptive-opinion.csv")

    # Convert labels: deceptive=1 (fake), truthful=0 (real)
    df['label'] = df['deceptive'].apply(lambda x: 1 if x == 'deceptive' else 0)
    df['text']  = df['text'].astype(str).str.strip()
    df['rating'] = 4.0  # OpSpam has no rating — use neutral default

    print(f"[opspam] Dataset: {len(df)} reviews")
    print(f"  Real (truthful) : {len(df[df.label==0])}")
    print(f"  Fake (deceptive): {len(df[df.label==1])}")

    # Extract features
    print("[opspam] Extracting features...")
    df = extract_features(df)

    # Load models
    print("[opspam] Loading models...")
    baseline  = pickle.load(open("outputs/model_baseline.pkl",  "rb"))
    gbm       = pickle.load(open("outputs/model_gbm.pkl",       "rb"))
    sentinel  = pickle.load(open("outputs/model_sentinel.pkl",  "rb"))
    scaler    = pickle.load(open("outputs/scaler.pkl",          "rb"))
    tfidf_vec = pickle.load(open("outputs/tfidf_vec.pkl",       "rb"))

    X_text   = df['text']
    X_feat   = scaler.transform(df[FEATURE_COLS].fillna(0))
    X_tfidf  = tfidf_vec.transform(X_text)
    X_hybrid = hstack([X_tfidf, csr_matrix(X_feat)])
    y_true   = df['label'].values

    results = []

    def evaluate(name, y_pred, y_prob):
        acc   = accuracy_score(y_true, y_pred)
        f1    = f1_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        mcc   = matthews_corrcoef(y_true, y_pred)
        auc   = roc_auc_score(y_true, y_prob)
        print(f"\n── {name} ──────────────────────────────────")
        print(f"  Accuracy    : {acc:.4f}")
        print(f"  F1 Score    : {f1:.4f}")
        print(f"  ROC-AUC     : {auc:.4f}")
        print(f"  Cohen Kappa : {kappa:.4f}")
        print(f"  MCC         : {mcc:.4f}")
        print(classification_report(y_true, y_pred, target_names=['Real','Fake']))
        return {"model": name, "dataset": "OpSpam",
                "accuracy": round(acc,4), "f1": round(f1,4),
                "auc": round(auc,4), "kappa": round(kappa,4), "mcc": round(mcc,4)}

    # Baseline
    y_pred = baseline.predict(X_text)
    y_prob = baseline.predict_proba(X_text)[:,1]
    results.append(evaluate("Baseline TF-IDF + LR", y_pred, y_prob))

    # GBM
    y_pred = gbm.predict(X_feat)
    y_prob = gbm.predict_proba(X_feat)[:,1]
    results.append(evaluate("Linguistic + GBM", y_pred, y_prob))

    # SENTINEL Hybrid
    y_pred = sentinel.predict(X_hybrid)
    y_prob = sentinel.predict_proba(X_hybrid)[:,1]
    results.append(evaluate("SENTINEL Hybrid", y_pred, y_prob))

    # DistilBERT
    try:
        from src.distilbert_predict import load_distilbert, predict_batch
        print("\n[opspam] Running DistilBERT inference (this takes ~5 mins on CPU)...")
        db_model, db_tokenizer, db_device = load_distilbert()
        texts = df['text'].tolist()
        db_probs = predict_batch(texts, db_model, db_tokenizer, db_device)
        db_preds = [1 if p > 0.5 else 0 for p in db_probs]
        results.append(evaluate("DistilBERT (fine-tuned)", db_preds, db_probs))
    except Exception as e:
        print(f"[opspam] DistilBERT skipped: {e}")

    # Summary
    print("\n── OpSpam Benchmark Summary ─────────────────────")
    df_results = pd.DataFrame(results)
    print(df_results[['model','accuracy','f1','auc','kappa','mcc']].to_string(index=False))

    # Compare with published SOTA
    print("\n── Comparison with Published SOTA on OpSpam ─────")
    sota = [
        {"model": "Ott et al. 2011 (SVM)",         "accuracy": 0.896, "f1": 0.895},
        {"model": "Li et al. 2014 (Neural)",        "accuracy": 0.886, "f1": 0.884},
        {"model": "Rashkin et al. 2017 (LSTM)",     "accuracy": 0.912, "f1": 0.910},
        {"model": "Best published (BERT-based)",    "accuracy": 0.946, "f1": 0.944},
    ]
    for s in sota:
        print(f"  {s['model']:<40} Acc={s['accuracy']} F1={s['f1']}")

    best_ours = max(results, key=lambda x: x['f1'])
    print(f"\n  Our best: {best_ours['model']:<33} Acc={best_ours['accuracy']} F1={best_ours['f1']}")

    if best_ours['f1'] >= 0.944:
        print("\n  ✓ BEATS state of the art on OpSpam!")
    elif best_ours['f1'] >= 0.912:
        print("\n  ✓ Competitive with top published results")
    else:
        print(f"\n  Note: {round((0.944 - best_ours['f1'])*100, 1)}% below SOTA — expected since we trained on Amazon, not hotel reviews")

    # Save
    Path("outputs").mkdir(exist_ok=True)
    df_results.to_csv("outputs/opspam_results.csv", index=False)
    json.dump(results, open("outputs/opspam_results.json","w"), indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    all_models = sota + results
    names = [m['model'] for m in all_models]
    f1s   = [m['f1']    for m in all_models]
    colors = ['#8B92A5'] * len(sota) + ['#00E5FF'] * len(results)
    bars = ax.barh(names, f1s, color=colors, height=0.6)
    ax.axvline(0.944, color='#FF4757', linewidth=1.5, linestyle='--', label='Published SOTA')
    ax.set_xlabel('F1 Score')
    ax.set_title('OpSpam Benchmark — Our Models vs Published SOTA\n(Zero-shot: trained on Amazon, tested on hotel reviews)')
    ax.legend()
    ax.set_xlim(0.5, 1.0)
    for bar, val in zip(bars, f1s):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("outputs/opspam_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[opspam] Chart saved -> outputs/opspam_comparison.png")
    print("Done!")

if __name__ == "__main__":
    run_opspam_eval()