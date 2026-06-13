# VerdadEira — Detector de Fake News
> Projeto Acadêmico | Verificação de notícias com ML + Google Fact Check API

## Estrutura do Projeto

```
fakenews_detector/
├── dataset/
│   ├── dataset_inicial.csv      ← Dataset com 40 notícias verificadas
│   └── gerar_dataset.py         ← Script para gerar mais dados via factcheckexplorer
├── modelo/
│   └── treinar_modelo.py        ← Treina Naive Bayes, Logistic Reg., Random Forest, SVM
├── backend/
│   └── app.py                   ← API Flask com Fact Check + ML
└── frontend/
    └── index.html               ← Interface web responsiva
```

## Como Executar

### 1. Instalar dependências

```bash
pip install flask flask-cors requests scikit-learn pandas joblib
```

### 2. Treinar o Modelo ML

```bash
cd modelo
python treinar_modelo.py
```

Isso gera `modelo_treinado.pkl` e `modelo_meta.json` com as métricas.

### 3. (Opcional) Gerar Dataset maior

```bash
pip install factcheckexplorer
cd dataset
python gerar_dataset.py
```

### 4. Configurar a API Google (opcional)

1. Acesse: https://toolbox.google.com/factcheck/apis
2. Crie uma chave de API
3. Defina a variável de ambiente:
   ```bash
   export GOOGLE_API_KEY="sua-chave-aqui"
   ```

### 5. Iniciar o Backend

```bash
cd backend
python app.py
```

Roda em: http://localhost:5000

### 6. Abrir o Frontend

Abra `frontend/index.html` no navegador.

> 💡 Sem backend: o frontend funciona em **modo demo** com classificação local via JavaScript.

---

## Endpoints da API

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/` | Status da API |
| POST | `/verificar` | Analisa uma afirmação |
| GET | `/dataset` | Retorna últimos 50 registros |
| GET | `/modelo/info` | Métricas do modelo treinado |

### Exemplo de chamada

```bash
curl -X POST http://localhost:5000/verificar \
  -H "Content-Type: application/json" \
  -d '{"afirmacao": "Urnas eletrônicas foram hackeadas"}'
```

---

## Fluxo do Sistema

```
Usuário → Frontend → Backend Flask
                          │
                    Google Fact Check API
                          │
                    ┌─ SIM ─→ Retorna resultado + salva no dataset
                    │
                    └─ NÃO ─→ Modelo ML (Naive Bayes / LogReg / RF / SVM)
                                    │
                              Resultado estimado + salva no dataset
```

## Métricas Avaliadas

- **Accuracy** — acertos sobre total de predições
- **Precision** — dos classificados como verdadeiro, quantos eram de fato
- **Recall** — dos reais verdadeiros, quantos o modelo acertou
- **F1-Score** — média harmônica entre precision e recall

## Algoritmos Implementados

| Algoritmo | Características |
|-----------|----------------|
| Naive Bayes | Rápido, bom para texto, probabilístico |
| Logistic Regression | Boa linha de base, interpretável |
| Random Forest | Robusto, menos propenso a overfitting |
| SVM (LinearSVC) | Eficiente com alta dimensionalidade |

O **melhor modelo** (maior F1-score) é salvo automaticamente.

---

## Observação Acadêmica

> Este sistema possui caráter experimental. A classificação por ML não garante veracidade absoluta das informações — é uma ferramenta de apoio à análise crítica de notícias.
