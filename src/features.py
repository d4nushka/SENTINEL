import re
import math
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

def word_count(text):
    return len(text.split())

def char_count(text):
    return len(text)

def avg_word_length(text):
    words = text.split()
    return sum(len(w) for w in words) / len(words) if words else 0

def exclamation_count(text):
    return text.count('!')

def question_count(text):
    return text.count('?')

def caps_ratio(text):
    return sum(1 for c in text if c.isupper()) / len(text) if text else 0

def type_token_ratio(text):
    words = text.lower().split()
    return len(set(words)) / len(words) if words else 0

def repetition_score(text):
    words = text.lower().split()
    if len(words) < 2: return 0
    counts = Counter(words)
    repeated = sum(1 for w, c in counts.items() if c > 1 and len(w) > 3)
    return repeated / len(counts)

def sentiment_score(text):
    pos = ['amazing','excellent','perfect','love','great','fantastic','outstanding',
           'brilliant','wonderful','superb','incredible','best','awesome','exceptional']
    neg = ['terrible','awful','horrible','worst','bad','poor','disappointing',
           'useless','broken','cheap','fake','waste','defective','return']
    t = text.lower()
    p = sum(1 for w in pos if w in t)
    n = sum(1 for w in neg if w in t)
    return (p - n) / (p + n + 1)

def vagueness_score(text):
    vague = ['everything','everyone','always','never','perfect','best','greatest',
             'amazing','incredible','unbelievable','absolutely','completely',
             'totally','literally','honestly','genuinely','definitely','certainly']
    t = text.lower()
    count = sum(1 for w in vague if w in t)
    return count / (len(text.split()) + 1)

def burstiness(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 2: return 0
    lengths = [len(s.split()) for s in sentences]
    mean = np.mean(lengths)
    std  = np.std(lengths)
    return std / (mean + 1e-9)

def perplexity_proxy(text):
    words = text.lower().split()
    if len(words) < 2: return 0
    freq = Counter(words)
    total = len(words)
    probs = [freq[w] / total for w in words]
    return -sum(p * math.log(p + 1e-9) for p in probs)

def avg_sentence_length(text):
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences: return 0
    return np.mean([len(s.split()) for s in sentences])

def punctuation_ratio(text):
    if not text: return 0
    punct = sum(1 for c in text if c in '.,!?;:')
    return punct / len(text)

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[features] Extracting features for {len(df)} reviews...")
    df = df.copy()
    df['word_count']        = df['text'].apply(lambda x: word_count(str(x)))
    df['char_count']        = df['text'].apply(lambda x: char_count(str(x)))
    df['avg_word_len']      = df['text'].apply(lambda x: avg_word_length(str(x)))
    df['exclamation_cnt']   = df['text'].apply(lambda x: exclamation_count(str(x)))
    df['question_cnt']      = df['text'].apply(lambda x: question_count(str(x)))
    df['caps_ratio']        = df['text'].apply(lambda x: caps_ratio(str(x)))
    df['type_token_ratio']  = df['text'].apply(lambda x: type_token_ratio(str(x)))
    df['repetition_score']  = df['text'].apply(lambda x: repetition_score(str(x)))
    df['sentiment_score']   = df['text'].apply(lambda x: sentiment_score(str(x)))
    df['vagueness_score']   = df['text'].apply(lambda x: vagueness_score(str(x)))
    df['burstiness']        = df['text'].apply(lambda x: burstiness(str(x)))
    df['perplexity_proxy']  = df['text'].apply(lambda x: perplexity_proxy(str(x)))
    df['avg_sent_len']      = df['text'].apply(lambda x: avg_sentence_length(str(x)))
    df['punct_ratio']       = df['text'].apply(lambda x: punctuation_ratio(str(x)))
    df['rating']            = pd.to_numeric(df['rating'], errors='coerce').fillna(3.0)
    print(f"[features] Done — {len(df.columns)} total columns")
    return df

FEATURE_COLS = ['word_count','char_count','avg_word_len','exclamation_cnt',
                'question_cnt','caps_ratio','type_token_ratio','repetition_score',
                'sentiment_score','vagueness_score','burstiness','perplexity_proxy',
                'avg_sent_len','punct_ratio','rating']

if __name__ == "__main__":
    for split in ['reviews','train','val','test']:
        path = f"data/{split}.csv"
        if not Path(path).exists():
            continue
        df = pd.read_csv(path)
        df = extract_features(df)
        df.to_csv(f"data/{split}_features.csv", index=False)
        print(f"Saved -> data/{split}_features.csv")

    print("\nFeature stats (train):")
    train = pd.read_csv("data/train_features.csv")
    cols = ['word_count','exclamation_cnt','caps_ratio','vagueness_score','sentiment_score']
    print(train.groupby('label')[cols].mean().round(3))
    print("\nDone! Run next: python src/train.py")