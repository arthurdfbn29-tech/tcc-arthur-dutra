# TCC — Segmentação de clientes e previsão de churn em uma marca de moda feminina de luxo

Arthur Dutra — Universidade Federal Fluminense (UFF)

Códigos do trabalho, organizados em um script por etapa da metodologia (Capítulo 3).
Os dados de clientes **não** estão neste repositório (LGPD e sigilo comercial):
a planilha extraída, as bases tratadas e os arquivos intermediários (`*.csv`, `*.xlsx`)
ficam apenas em ambiente local e são bloqueados pelo `.gitignore`.

| Arquivo | Etapa |
|---|---|
| `query_anonimizada - Tcc.sql` | 2 — extração da base ativa de 2024 (versão anonimizada) |
| `limpeza_base.py` | 3 — validação, limpeza e geração das bases total e sem outliers |
| `aed.py` | 4 — análise exploratória |
| `etapa5a.py` | 5A — transformação e padronização das variáveis de segmentação |
| `etapa6a1.py` | 6A.1 — K-means, cotovelo (SSE) e silhueta |
| `etapa6a2.py` | 6A.2 — DBSCAN e gráfico k-dist |
| `etapa7a.py` | 7A — comparação e perfil dos clusters |
| `etapa5b.py` | 5B — churn, Information Value e divisão treino/teste 70/30 |
| `etapa6b1.py` | 6B.1 — regressão logística (statsmodels) |
| `etapa6b2.py` | 6B.2 — árvore de decisão, poda e validação cruzada |
| `etapa7b.py` | 7B — avaliação no conjunto de teste e bootstrap pareado |
| `etapa8.py` | 8 — integração clusters × churn × modelos |

Execução: rodar os scripts na ordem da tabela; o cabeçalho de cada um descreve entradas,
saídas e argumentos. `limpeza_base.py` exige a variável de ambiente `TCC_SAL` (sal do hash
do identificador da cliente). Os números citados no texto ficam em `numeros_etapa*.txt` e
as figuras em `Figuras/`. Todas as etapas com componente aleatório usam semente fixa.

Ambiente: Python 3.10, pandas 2.3, NumPy 2.2, Matplotlib 3.10, scikit-learn 1.7, statsmodels 0.15.
