"""
Treinamento do modelo de Machine Learning para detecção de fake news.
Execute: pip install scikit-learn pandas nltk joblib
"""

import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

# ── 1. Carregar Dataset ───────────────────────────────────────────────────────

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../dataset/dataset_inicial.csv")

df = pd.read_csv(DATASET_PATH)
print(f"Dataset carregado: {len(df)} entradas")
print(df["label"].value_counts())

# ── 2. Pré-processamento de texto ─────────────────────────────────────────────

def preprocessar(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúâêîôûãõàèìòùäëïöüç\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

df["text_clean"] = df["text"].apply(preprocessar)
df = df[df["text_clean"].str.len() > 5]

X = df["text_clean"]
y = (df["label"] == "verdadeiro").astype(int)  # 1=verdadeiro, 0=falso

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTreino: {len(X_train)} | Teste: {len(X_test)}")

# ── 3. Modelos ────────────────────────────────────────────────────────────────

modelos = {
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf", MultinomialNB())
    ]),
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    "SVM": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf", LinearSVC(random_state=42, max_iter=2000))
    ]),
}

# ── 4. Treinamento e Avaliação ────────────────────────────────────────────────

resultados = {}
melhor_modelo = None
melhor_f1 = 0

print("\n" + "="*60)
print("RESULTADOS DOS MODELOS")
print("="*60)

for nome, pipeline in modelos.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    resultados[nome] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    print(f"\n📊 {nome}")
    print(f"   Accuracy : {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}")
    print(f"   F1-Score : {f1:.4f}")

    if f1 > melhor_f1:
        melhor_f1 = f1
        melhor_modelo = (nome, pipeline)

# ── 5. Salvar o melhor modelo ─────────────────────────────────────────────────

nome_melhor, pipeline_melhor = melhor_modelo
print(f"\n✅ Melhor modelo: {nome_melhor} (F1={melhor_f1:.4f})")

modelo_path = os.path.join(os.path.dirname(__file__), "modelo_treinado.pkl")
joblib.dump(pipeline_melhor, modelo_path)
print(f"Modelo salvo em: {modelo_path}")

# Salva metadados
import json
meta = {
    "melhor_modelo": nome_melhor,
    "metricas": resultados[nome_melhor],
    "todos_resultados": resultados,
    "dataset_size": len(df),
    "features": "TF-IDF unigram+bigram, max_features=5000"
}
meta_path = os.path.join(os.path.dirname(__file__), "modelo_meta.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"Metadados salvos em: {meta_path}")
print("\nTreinamento concluído!")
