"""
Backend Flask - Detector de Fake News
Instale: pip install flask flask-cors requests scikit-learn pandas joblib
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import joblib
import os
import re
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Permite chamadas do frontend

# ── Caminhos ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH  = os.path.join(BASE_DIR, "../modelo/modelo_treinado.pkl")
META_PATH    = os.path.join(BASE_DIR, "../modelo/modelo_meta.json")
DATASET_PATH = os.path.join(BASE_DIR, "../dataset/dataset_inicial.csv")

# ── Configurações ─────────────────────────────────────────────────────────────
GOOGLE_FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
# Cole sua chave aqui ou defina a variável de ambiente GOOGLE_API_KEY
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "SUA_CHAVE_AQUI")

# ── Carrega modelo ML ─────────────────────────────────────────────────────────
modelo = None
modelo_info = {}

def carregar_modelo():
    global modelo, modelo_info
    if os.path.exists(MODELO_PATH):
        modelo = joblib.load(MODELO_PATH)
        print("✅ Modelo ML carregado.")
    else:
        print("⚠️  Modelo não encontrado. Execute modelo/treinar_modelo.py primeiro.")

    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            modelo_info = json.load(f)

carregar_modelo()

# ── Helpers ───────────────────────────────────────────────────────────────────

def preprocessar(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúâêîôûãõàèìòùäëïöüç\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def salvar_no_dataset(texto, label, fonte, rating, url=""):
    """Append de novo registro ao CSV local."""
    novo = pd.DataFrame([{
        "text": texto, "label": label,
        "fonte": fonte, "rating": rating, "url": url,
        "data_coleta": datetime.now().isoformat()
    }])
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        df = pd.concat([df, novo], ignore_index=True)
    else:
        df = novo
    df.to_csv(DATASET_PATH, index=False, encoding="utf-8")


def consultar_google_factcheck(query):
    """Consulta a API Google Fact Check."""
    if GOOGLE_API_KEY == "SUA_CHAVE_AQUI":
        return None  # API não configurada

    try:
        params = {"query": query, "key": GOOGLE_API_KEY, "languageCode": "pt"}
        resp = requests.get(GOOGLE_FACTCHECK_URL, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            claims = data.get("claims", [])
            if claims:
                claim = claims[0]
                review = claim.get("claimReview", [{}])[0]
                return {
                    "texto_original": claim.get("text", ""),
                    "publisher": review.get("publisher", {}).get("name", ""),
                    "rating": review.get("textualRating", ""),
                    "url": review.get("url", ""),
                    "titulo": review.get("title", ""),
                }
    except Exception as e:
        print(f"Erro na API Google: {e}")
    return None


def classificar_com_ml(texto):
    """Classifica usando o modelo local."""
    if modelo is None:
        return None, 0.0
    texto_limpo = preprocessar(texto)
    pred = modelo.predict([texto_limpo])[0]
    # Tenta obter probabilidade
    try:
        prob = modelo.predict_proba([texto_limpo])[0]
        confianca = float(max(prob))
    except Exception:
        confianca = 0.75  # valor padrão para modelos sem predict_proba
    label = "verdadeiro" if pred == 1 else "falso"
    return label, confianca

# ── Rotas ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "API Fake News Detector rodando!"})


@app.route("/verificar", methods=["POST"])
def verificar():
    """
    Recebe { "afirmacao": "texto..." } e retorna análise.
    """
    data = request.get_json()
    if not data or "afirmacao" not in data:
        return jsonify({"erro": "Campo 'afirmacao' obrigatório"}), 400

    afirmacao = data["afirmacao"].strip()
    if len(afirmacao) < 5:
        return jsonify({"erro": "Afirmação muito curta"}), 400

    # 1. Tenta API Google Fact Check
    resultado_google = consultar_google_factcheck(afirmacao)

    if resultado_google:
        rating_raw = resultado_google["rating"].lower()
        if any(w in rating_raw for w in ["verdadeiro", "true", "correto", "fato", "verdad"]):
            label = "verdadeiro"
        else:
            label = "falso"

        salvar_no_dataset(
            texto=afirmacao,
            label=label,
            fonte=resultado_google["publisher"],
            rating=resultado_google["rating"],
            url=resultado_google["url"]
        )

        return jsonify({
            "afirmacao": afirmacao,
            "fonte": "Google Fact Check API",
            "resultado": label,
            "rating": resultado_google["rating"],
            "publisher": resultado_google["publisher"],
            "url": resultado_google["url"],
            "confianca": None,
            "metodo": "api"
        })

    # 2. Fallback: Modelo ML
    label_ml, confianca = classificar_com_ml(afirmacao)

    if label_ml is None:
        return jsonify({
            "afirmacao": afirmacao,
            "resultado": "indefinido",
            "mensagem": "Modelo ML não disponível. Execute treinar_modelo.py.",
            "metodo": "nenhum"
        }), 503

    salvar_no_dataset(
        texto=afirmacao,
        label=label_ml,
        fonte="Modelo ML Local",
        rating=f"{label_ml} ({confianca:.0%})"
    )

    return jsonify({
        "afirmacao": afirmacao,
        "fonte": "Modelo de Machine Learning",
        "resultado": label_ml,
        "rating": f"Classificado como {label_ml}",
        "confianca": round(confianca, 4),
        "modelo_usado": modelo_info.get("melhor_modelo", "desconhecido"),
        "metodo": "ml"
    })


@app.route("/dataset", methods=["GET"])
def get_dataset():
    """Retorna os últimos 50 registros do dataset."""
    if not os.path.exists(DATASET_PATH):
        return jsonify({"dados": [], "total": 0})
    df = pd.read_csv(DATASET_PATH)
    registros = df.tail(50).to_dict(orient="records")
    return jsonify({"dados": registros, "total": len(df)})


@app.route("/modelo/info", methods=["GET"])
def get_modelo_info():
    """Retorna informações sobre o modelo treinado."""
    return jsonify(modelo_info if modelo_info else {"erro": "Modelo não treinado"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
