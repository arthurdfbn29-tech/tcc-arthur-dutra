"""
Etapa 5A - Engenharia de variaveis para segmentacao (Frente A) do TCC.

Constroi o espaco de agrupamento usado pelo K-means (Etapa 6A.1) e pelo
DBSCAN (Etapa 6A.2), aplicando os tratamentos descritos na subsecao de
pre-processamento do Capitulo 2:

  1. Recria a recencia (mesma definicao da Etapa 4: 31/12/2024 - ultimo pedido).
  2. Verifica ausentes e positividade (o ln exige valores > 0).
  3. Aplica ln em Frequencia_total e TM_total (assimetria elevada na AED).
     A recencia NAO recebe log (assimetria baixa e pode valer zero).
  4. Padroniza as tres variaveis por escore-z, com media e desvio-padrao
     calculados DENTRO de cada base (base_total e base_sem_outliers sao
     experimentos independentes).
  5. Verifica o resultado: assimetria antes/depois, media/desvio apos o
     escore-z, correlacao no espaco final e a identidade ln VA = ln F + ln TM.

Uso:  python etapa5a.py [pasta_das_bases]

Saidas:
  numeros_etapa5a.txt                 -> numeros citados no texto, para conferencia
  Figuras/etapa5a_transformacoes.png  -> distribuicoes antes/depois do log
  matriz_5a_base_total.csv            -> espaco padronizado (GITIGNORED: contem hash)
  matriz_5a_base_sem_outliers.csv     -> idem

Convencoes:
  - ln = logaritmo natural.
  - Escore-z com desvio-padrao populacional (ddof=0), mesma convencao do
    StandardScaler do scikit-learn.
  - Assimetria: estimador ajustado de Fisher-Pearson (pandas .skew()),
    o mesmo usado na Etapa 4.

NAO versionar os CSVs (contem identificador derivado de CPF).
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- Paleta do projeto (mesma do aed.py) -----------------------------------
COR_PRIMARIA = "#1C1C1C"
COR_OURO     = "#B8964F"
COR_BORDO    = "#6B2737"
COR_CINZA    = "#9A9A9A"
COR_OSSO     = "#DCD6CC"

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.edgecolor": COR_CINZA, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": COR_OSSO, "grid.linewidth": 0.6,
    "grid.alpha": 0.7, "axes.axisbelow": True, "legend.frameon": False,
    "xtick.color": COR_PRIMARIA, "ytick.color": COR_PRIMARIA,
    "text.color": COR_PRIMARIA, "axes.labelcolor": COR_PRIMARIA,
})

PASTA = sys.argv[1] if len(sys.argv) > 1 else "."
FIGS = os.path.join(PASTA, "Figuras")
os.makedirs(FIGS, exist_ok=True)
DATA_CORTE = pd.Timestamp("2024-12-31")
BASES = ["base_total", "base_sem_outliers"]

# Variaveis do espaco de agrupamento e se recebem log
ESPACO = [("recencia", False), ("Frequencia_total", True), ("TM_total", True)]
NOMES = {"recencia": "Recência", "Frequencia_total": "Frequência",
         "TM_total": "Ticket médio"}

_saida = []
def reg(txt=""):
    print(txt); _saida.append(str(txt))

def milhar(v, _p=None):
    return f"{v:,.0f}".replace(",", ".")


def carregar(nome):
    df = pd.read_csv(os.path.join(PASTA, f"{nome}.csv"),
                     parse_dates=["data_ultimo_pedido_2024"])
    df["recencia"] = (DATA_CORTE - df.data_ultimo_pedido_2024).dt.days
    return df


def construir_espaco(df, nome):
    reg("=" * 78); reg(f"BASE: {nome}  ({len(df):,} clientes)".replace(",", ".")); reg("=" * 78)

    # ---- 1. Checagens ---------------------------------------------------
    reg("[1] Checagens")
    cols = [c for c, _ in ESPACO] + ["VA_bruto_total"]
    for c in cols:
        reg(f"    {c:<18} ausentes={df[c].isna().sum():>6}  "
            f"min={df[c].min():>12.4f}  <=0: {(df[c] <= 0).sum():>6}")
    reg(f"    recencia == 0 (compra em 31/12): {(df.recencia == 0).sum()}")
    for c, usa_log in ESPACO:
        if usa_log:
            assert (df[c] > 0).all(), f"{c} tem valores <= 0: ln indefinido"
    assert df[cols].notna().all().all(), "ha valores ausentes no espaco"

    # ---- 2. Transformacao logaritmica -----------------------------------
    X = pd.DataFrame(index=df.index)
    for c, usa_log in ESPACO:
        X[c] = np.log(df[c]) if usa_log else df[c].astype(float)

    reg("[2] Assimetria antes e depois do ln")
    reg(f"    {'variavel':<18}{'original':>10}{'apos ln':>10}")
    assim = {}
    for c, usa_log in ESPACO:
        a0 = df[c].skew(); a1 = X[c].skew() if usa_log else np.nan
        assim[c] = (a0, a1)
        reg(f"    {c:<18}{a0:>10.2f}{(f'{a1:.2f}' if usa_log else '--'):>10}")
    a_va = (df.VA_bruto_total.skew(), np.log(df.VA_bruto_total).skew())
    reg(f"    {'VA_bruto_total':<18}{a_va[0]:>10.2f}{a_va[1]:>10.2f}  (referencia; fora do espaco)")

    # ---- 3. Escore-z (parametros da propria base) -----------------------
    mu = X.mean(); sd = X.std(ddof=0)
    Z = (X - mu) / sd
    reg("[3] Parametros do escore-z (escala apos ln, quando aplicavel)")
    for c, _ in ESPACO:
        reg(f"    {c:<18} media={mu[c]:>10.4f}  desvio={sd[c]:>10.4f}  "
            f"| conferencia: media_z={Z[c].mean():+.1e} desvio_z={Z[c].std(ddof=0):.4f}")
    reg(f"    amplitude z: " + "  ".join(
        f"{c}=[{Z[c].min():.2f}; {Z[c].max():.2f}]" for c, _ in ESPACO))

    # ---- 4. Correlacoes no espaco final ---------------------------------
    reg("[4] Correlacao no espaco final")
    pear = Z.corr(method="pearson"); spear = Z.corr(method="spearman")
    reg("    Pearson (relevante para a distancia euclidiana):")
    reg("    " + pear.round(3).to_string().replace("\n", "\n    "))
    reg("    Spearman:")
    reg("    " + spear.round(3).to_string().replace("\n", "\n    "))
    reg(f"    Pearson max |r| fora da diagonal: "
        f"{(pear.abs().values[~np.eye(3, dtype=bool)]).max():.3f}")

    # ---- 5. Identidade ln VA = ln F + ln TM -----------------------------
    resid = np.log(df.VA_bruto_total) - (np.log(df.Frequencia_total) + np.log(df.TM_total))
    reg("[5] Identidade ln VA = ln F + ln TM")
    reg(f"    |residuo| max = {resid.abs().max():.2e}   "
        f"clientes com |residuo| > 1e-6: {(resid.abs() > 1e-6).sum()}")
    r_va = np.corrcoef(np.log(df.VA_bruto_total),
                       np.log(df.Frequencia_total) + np.log(df.TM_total))[0, 1]
    reg(f"    Pearson(ln VA, ln F + ln TM) = {r_va:.6f}")
    reg(f"    Pearson(ln VA, ln F) = {np.corrcoef(np.log(df.VA_bruto_total), X.Frequencia_total)[0,1]:.3f}"
        f"   Pearson(ln VA, ln TM) = {np.corrcoef(np.log(df.VA_bruto_total), X.TM_total)[0,1]:.3f}")

    # ---- 6. Estrutura discreta (relevante para o DBSCAN) ----------------
    reg("[6] Estrutura discreta do espaco")
    f1 = (df.Frequencia_total == 1).mean()
    reg(f"    clientes com F = 1 (ln F = 0): {(df.Frequencia_total == 1).sum():,} "
        f"({f1*100:.1f}%)".replace(",", "."))
    reg(f"    valores distintos: recencia={df.recencia.nunique()}  "
        f"F={df.Frequencia_total.nunique()}  TM={df.TM_total.nunique()}")
    dup = Z.round(10).duplicated(keep=False).sum()
    reg(f"    pontos distintos no espaco 3D: {len(Z.round(10).drop_duplicates()):,}"
        f"   clientes em pontos repetidos: {dup:,} ({dup/len(Z)*100:.1f}%)".replace(",", "."))

    # ---- Salva matriz ----------------------------------------------------
    out = pd.DataFrame({"ClienteID": df.ClienteID,
                        "z_recencia": Z.recencia,
                        "z_ln_frequencia": Z.Frequencia_total,
                        "z_ln_ticket": Z.TM_total})
    caminho = os.path.join(PASTA, f"matriz_5a_{nome}.csv")
    out.to_csv(caminho, index=False)
    reg(f"    [matriz] {os.path.basename(caminho)}  ({out.shape[0]} x {out.shape[1]})")
    reg()
    return X, Z, assim


resultados = {}
for b in BASES:
    df = carregar(b)
    resultados[b] = (df,) + construir_espaco(df, b)

# ---------------------------------------------------------------------------
# Figura: distribuicoes antes/depois do ln (base total)
# ---------------------------------------------------------------------------
df, X, Z, assim = resultados["base_total"]
fig, ax = plt.subplots(2, 2, figsize=(9, 5.6))

# (a) Frequencia original (truncada no p99 so para visualizacao)
p99 = df.Frequencia_total.quantile(.99)
vc = df.Frequencia_total[df.Frequencia_total <= p99].value_counts().sort_index()
ax[0, 0].bar(vc.index, vc.values, color=COR_PRIMARIA, width=.8)
ax[0, 0].set_title(f"(a) Frequência original (assimetria = {assim['Frequencia_total'][0]:.2f})".replace(".", ","))
ax[0, 0].set_xlabel(f"Pedidos no ano (exibido até o p99 = {p99:.0f})")
ax[0, 0].set_ylabel("Clientes")

# (b) ln Frequencia
vcl = np.log(df.Frequencia_total).value_counts().sort_index()
ax[0, 1].bar(vcl.index, vcl.values, color=COR_OURO, width=.06)
ax[0, 1].set_title(f"(b) ln(frequência) (assimetria = {assim['Frequencia_total'][1]:.2f})".replace(".", ","))
ax[0, 1].set_xlabel("ln(pedidos no ano)")
ax[0, 1].set_ylabel("Clientes")

# (c) Ticket original (truncado no p99)
p99t = df.TM_total.quantile(.99)
ax[1, 0].hist(df.TM_total[df.TM_total <= p99t], bins=60, color=COR_PRIMARIA)
ax[1, 0].set_title(f"(c) Ticket médio original (assimetria = {assim['TM_total'][0]:.2f})".replace(".", ","))
ax[1, 0].set_xlabel(f"R\\$ por pedido (exibido até o p99 = R\\$ {milhar(p99t)})")
ax[1, 0].set_ylabel("Clientes")
ax[1, 0].xaxis.set_major_formatter(FuncFormatter(milhar))

# (d) ln Ticket
ax[1, 1].hist(X.TM_total, bins=60, color=COR_OURO)
ax[1, 1].set_title(f"(d) ln(ticket médio) (assimetria = {assim['TM_total'][1]:.2f})".replace(".", ","))
ax[1, 1].set_xlabel("ln(R\\$ por pedido)")
ax[1, 1].set_ylabel("Clientes")

for a in ax.flat:
    a.yaxis.set_major_formatter(FuncFormatter(milhar))
fig.suptitle("Efeito da transformação logarítmica (base total)", y=1.01, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "etapa5a_transformacoes.png"), facecolor="white")
plt.close(fig)
reg("[figura] etapa5a_transformacoes.png")

with open(os.path.join(PASTA, "numeros_etapa5a.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(_saida) + "\n")
