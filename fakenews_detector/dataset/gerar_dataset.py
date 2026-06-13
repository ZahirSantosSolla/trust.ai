"""
Script para geração do dataset inicial usando factcheckexplorer.
Execute: pip install factcheckexplorer pandas
"""

import pandas as pd
import os
import json

# Tenta usar factcheckexplorer; se não estiver instalado, usa o CSV local
try:
    from factcheckexplorer.factcheckexplorer import FactCheckLib

    KEYWORDS = [
        "eleição", "bolsonaro", "lula", "pt",
        "campanha", "urna", "voto", "fraude"
    ]

    all_data = []

    for keyword in KEYWORDS:
        print(f"Coletando dados para: {keyword}")
        try:
            fact_check = FactCheckLib(
                query=keyword,
                language="pt",
                num_results=100
            )
            result = fact_check.process()

            if result:
                for item in result:
                    try:
                        text = item.get("text", "")
                        rating = item.get("textualRating", "").lower()
                        publisher = item.get("publisher", {}).get("name", "")
                        url = item.get("url", "")

                        # Normaliza o label
                        label = "falso"
                        if any(w in rating for w in ["verdadeiro", "true", "correto", "fato"]):
                            label = "verdadeiro"

                        if text:
                            all_data.append({
                                "text": text,
                                "label": label,
                                "fonte": publisher,
                                "rating": rating,
                                "url": url,
                                "keyword": keyword
                            })
                    except Exception as e:
                        print(f"Erro ao processar item: {e}")
        except Exception as e:
            print(f"Erro na keyword '{keyword}': {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        df = df.drop_duplicates(subset=["text"])
        df.to_csv("dataset_coletado.csv", index=False, encoding="utf-8")
        print(f"\nDataset gerado com {len(df)} entradas!")
        print(df["label"].value_counts())
    else:
        print("Nenhum dado coletado. Usando dataset local.")

except ImportError:
    print("factcheckexplorer não instalado.")
    print("Use: pip install factcheckexplorer")
    print("Usando dataset_inicial.csv como base.")
    df = pd.read_csv("dataset_inicial.csv")
    print(f"Dataset local: {len(df)} entradas")
    print(df["label"].value_counts())
