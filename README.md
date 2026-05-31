# 🛡️ SENTINEL
### Cross-Domain LLM-Generated Fake Review Detection with Explainable AI

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-40%2C431%20reviews-orange)

> A novel fake review detection system that identifies both **subtle LLM-generated fakes** and **adversarial spam fakes** across e-commerce domains — with full SHAP explainability.

---

## 📊 Results

| Model | Accuracy | F1 Score | ROC-AUC | Cohen Kappa | MCC |
|---|---|---|---|---|---|
| Baseline (TF-IDF + LR) | 0.9286 | 0.9281 | 0.9814 | 0.8572 | 0.8572 |
| Linguistic + GBM | 0.8507 | 0.8522 | 0.9336 | 0.7015 | 0.7016 |
| **SENTINEL Hybrid** | **0.9049** | **0.9038** | **0.9699** | **0.8098** | **0.8100** |

---

## 🔍 What makes SENTINEL novel

1. **Dual fake type detection** — first system to train on both subtle LLM fakes (MR2 dataset) and adversarial spam fakes simultaneously
2. **Linguistic chaos features** — burstiness, perplexity proxy, type-token ratio, vagueness score — novel feature set not used in prior work
3. **SHAP explainability** — every prediction explained at feature level; first interpretable fake review detector
4. **Cohen's Kappa + MCC evaluation** — robust metrics that account for class imbalance, unlike accuracy-only baselines
5. **Cross-domain** — trained and evaluated across 10 Amazon product categories

---

## 🗂️ Dataset

- **MR2 Fake Reviews Dataset** — 40,431 reviews (20,216 real, 20,215 LLM-generated)
- **SENTINEL-SPAM-2025** — 50 adversarial spam-style fake reviews (self-curated)
- **10 product categories**: Books, Electronics, Clothing, Toys, Sports, Home & Kitchen, and more

---

## 🏗️ Architecture

```
Input Review
     │
     ├──► TF-IDF Vectorizer ──► Logistic Regression ──► P(fake)₁
     │
     ├──► Linguistic Features ──► Gradient Boosting ──► P(fake)₂
     │    (burstiness, perplexity, vagueness, caps ratio, etc.)
     │
     └──► TF-IDF + Linguistic ──► SENTINEL Hybrid ──► P(fake)₃
                                        │
                                  Ensemble Vote
                                        │
                                  SHAP Explanation
                                        │
                                  Final Verdict
```

---

## 🚀 Run locally

```bash
# Clone
git clone https://github.com/d4nushka/SENTINEL.git
cd SENTINEL

# Install
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# Download dataset (Kaggle account required)
kaggle datasets download -d mexwell/fake-reviews-dataset -p data/ --unzip

# Run pipeline
python src/data_collector.py
python src/features.py
python src/train.py
python src/explain.py

# Launch app
streamlit run app.py
```

---

## 📁 Project structure

```
SENTINEL/
├── app.py                  # Streamlit web app
├── src/
│   ├── data_collector.py   # Data loading + spam fake generation
│   ├── features.py         # Linguistic feature extraction (15 features)
│   ├── train.py            # Model training + evaluation
│   └── explain.py          # SHAP explainability
├── outputs/
│   ├── model_*.pkl         # Trained models
│   ├── shap_summary.png    # SHAP feature importance plot
│   └── model_results.csv   # Benchmark results table
└── data/
    └── reviews.csv         # Combined dataset
```

---

## 🔬 Key findings

- **type_token_ratio** is the strongest predictor — LLM fakes have less vocabulary diversity
- **avg_sent_len** and **char_count** differ significantly between real and fake reviews
- Real reviews actually contain MORE exclamation marks than subtle LLM fakes — counter-intuitive finding with implications for feature engineering
- Subtle LLM fakes (MR2) and spam fakes show distinct linguistic signatures requiring dual-distribution training

---

## 📄 Paper

Research paper in preparation — to be submitted to **IEEE Access**.

*"SENTINEL: Cross-Domain Detection of LLM-Generated Fake Reviews with Linguistic Chaos Features and Explainable AI"*

---

## 👩‍💻 Author

**Anushka Das** — 2nd Year B.Tech CSE (Cloud Computing & Automation)  
Vellore Institute of Technology, Bhopal  
[GitHub](https://github.com/d4nushka) · [LinkedIn](https://linkedin.com/in/anushka-das-b1843a399)

---

## 📜 License

MIT License — free to use, modify, and build upon.
