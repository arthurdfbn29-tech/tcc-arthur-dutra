"""
Etapa 4 - Analise Exploratoria dos Dados (AED) do TCC.

Le as duas bases analiticas produzidas pela Etapa 3 (limpeza_base.py),
deriva as variaveis que a AED precisa (recencia, faixas, RFM) e gera
TODAS as figuras e tabelas citadas no texto da Etapa 4.

Uso:  python aed.py [pasta_das_bases]
      (default: pasta atual; espera base_total.csv e base_sem_outliers.csv)

Saidas:
  Figuras/aed_*.png      -> figuras para o LaTeX (template: \\includegraphics)
  numeros_etapa4.txt     -> todos os numeros citados no texto, para conferencia

Premissas registradas:
  - Data de corte para recencia: 31/12/2024 (ultimo dia da janela de observacao).
  - churn refere-se a 2025; todas as preditoras descrevem 2024. Nao ha
    sobreposicao entre janela de observacao e janela de resultado.

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

# ---------------------------------------------------------------------------
# Configuracao visual - paleta sobria, coerente com posicionamento de LUXO
# ---------------------------------------------------------------------------
COR_PRIMARIA = "#1C1C1C"   # preto suave (texto, barras principais)
COR_OURO     = "#B8964F"   # champagne (destaque)
COR_BORDO    = "#6B2737"   # bordo (alerta / churn)
COR_CINZA    = "#9A9A9A"   # cinza medio (secundario)
COR_OSSO     = "#DCD6CC"   # osso (fundo de barra / grade)

PALETA_GRUPOS = {
    "Normal":           COR_CINZA,
    "Outlier Superior": COR_OURO,
    "Outlier Inferior": COR_BORDO,
}

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.edgecolor": COR_CINZA,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": COR_OSSO,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.7,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "xtick.color": COR_PRIMARIA,
    "ytick.color": COR_PRIMARIA,
    "text.color": COR_PRIMARIA,
    "axes.labelcolor": COR_PRIMARIA,
})

PASTA = sys.argv[1] if len(sys.argv) > 1 else "."
FIGS = os.path.join(PASTA, "Figuras")
os.makedirs(FIGS, exist_ok=True)
DATA_CORTE = pd.Timestamp("2024-12-31")

_saida = []


def reg(txt=""):
    """Registra uma linha tanto no stdout quanto no arquivo de conferencia."""
    print(txt)
    _saida.append(str(txt))


def brl(v, _p=None):
    """Formata em Real brasileiro (milhar com ponto, decimal com virgula)."""
    return ("R$ " + f"{v:,.2f}").replace(",", "~").replace(".", ",").replace("~", ".")


def milhar(v, _p=None):
    return f"{v:,.0f}".replace(",", ".")


def salvar(fig, nome):
    caminho = os.path.join(FIGS, nome)
    fig.savefig(caminho, facecolor="white")
    plt.close(fig)
    reg(f"    [figura] {nome}")


# ---------------------------------------------------------------------------
# 1. Carga e variaveis derivadas
# ---------------------------------------------------------------------------
DATAS = ["data_1_pedido_2024", "data_ultimo_pedido_2024",
         "data_1_pedido_2023", "data_ultimo_pedido_2023"]

def carregar(nome):
    df = pd.read_csv(os.path.join(PASTA, f"{nome}.csv"), parse_dates=DATAS)
    # --- Recencia: dias entre o ultimo pedido de 2024 e a data de corte.
    #     Quanto MAIOR, mais tempo o cliente esta inativo.
    df["recencia"] = (DATA_CORTE - df.data_ultimo_pedido_2024).dt.days
    # --- Tempo de relacionamento observado dentro de 2024
    df["tempo_vida_2024"] = (df.data_ultimo_pedido_2024 - df.data_1_pedido_2024).dt.days
    df["mes_1o_pedido"] = df.data_1_pedido_2024.dt.month
    df["mes_ultimo_pedido"] = df.data_ultimo_pedido_2024.dt.month
    # --- Faixas para analise bivariada
    df["faixa_freq"] = pd.cut(df.Frequencia_total, [0, 1, 2, 3, 5, 10, np.inf],
                              labels=["1", "2", "3", "4-5", "6-10", "11+"])
    df["faixa_recencia"] = pd.cut(df.recencia, [-1, 30, 90, 180, 270, 366],
                                  labels=["0-30", "31-90", "91-180", "181-270", "271-365"])
    df["faixa_va"] = pd.qcut(df.VA_bruto_total, 5,
                             labels=["Q1 (menor)", "Q2", "Q3", "Q4", "Q5 (maior)"])
    idade_ok = df.idade_valida == 1
    df["faixa_idade"] = np.where(idade_ok,
        pd.cut(df.idade_fim_2024, [0, 25, 35, 45, 55, 100],
               labels=["até 25", "26-35", "36-45", "46-55", "56+"]).astype(object),
        "sem informação")
    return df

reg("=" * 78)
reg("ETAPA 4 - ANALISE EXPLORATORIA DOS DADOS")
reg("=" * 78)
reg(f"Data de corte para recencia: {DATA_CORTE.date()}")
reg()

bt = carregar("base_total")
bs = carregar("base_sem_outliers")
reg(f"base_total      : {bt.shape[0]:>6} clientes x {bt.shape[1]} colunas (com derivadas)")
reg(f"base_sem_outliers: {bs.shape[0]:>6} clientes x {bs.shape[1]} colunas (com derivadas)")
reg()

# ---------------------------------------------------------------------------
# 2. Panorama geral - os "big numbers"
# ---------------------------------------------------------------------------
reg("-" * 78)
reg("[1] PANORAMA GERAL")
reg("-" * 78)

def panorama(df, nome):
    return {
        "base": nome,
        "clientes": len(df),
        "receita": df.VA_bruto_total.sum(),
        "ticket_mediano": df.TM_total.median(),
        "va_mediano": df.VA_bruto_total.median(),
        "freq_mediana": df.Frequencia_total.median(),
        "freq_media": df.Frequencia_total.mean(),
        "recencia_mediana": df.recencia.median(),
        "churn": df.churn.mean() * 100,
        "multicanal": (df.canais_cliente == "Multicanal").mean() * 100,
        "estados": df.estado.nunique(),
    }

pan = pd.DataFrame([panorama(bt, "base_total"), panorama(bs, "base_sem_outliers")])
for _, r in pan.iterrows():
    reg(f"  {r['base']}")
    reg(f"    Clientes ................ {milhar(r['clientes'])}")
    reg(f"    Receita total ........... {brl(r['receita'])}")
    reg(f"    VA mediano por cliente .. {brl(r['va_mediano'])}")
    reg(f"    Ticket medio (mediana) .. {brl(r['ticket_mediano'])}")
    reg(f"    Frequencia mediana ...... {r['freq_mediana']:.0f} (media {r['freq_media']:.2f})")
    reg(f"    Recencia mediana ........ {r['recencia_mediana']:.0f} dias")
    reg(f"    Taxa de churn ........... {r['churn']:.1f}%")
    reg(f"    Multicanal .............. {r['multicanal']:.1f}%")
    reg(f"    Estados representados ... {r['estados']}")
    reg()

# ---------------------------------------------------------------------------
# 3. Estatisticas descritivas das variaveis numericas
# ---------------------------------------------------------------------------
reg("-" * 78)
reg("[2] ESTATISTICAS DESCRITIVAS (base_total)")
reg("-" * 78)
NUMERICAS = ["VA_bruto_total", "Frequencia_total", "TM_total", "recencia", "idade_fim_2024"]
desc = []
for c in NUMERICAS:
    s = bt[c].dropna() if c != "idade_fim_2024" else bt.loc[bt.idade_valida == 1, c]
    desc.append({
        "variavel": c, "n": len(s), "media": s.mean(), "dp": s.std(),
        "cv": s.std() / s.mean(), "min": s.min(), "p25": s.quantile(.25),
        "mediana": s.median(), "p75": s.quantile(.75), "p90": s.quantile(.90),
        "p99": s.quantile(.99), "max": s.max(), "assimetria": s.skew(),
        "curtose": s.kurtosis(),
    })
desc = pd.DataFrame(desc).set_index("variavel")
reg(desc.round(2).to_string())
reg()
reg("  Leitura: assimetria alta em VA/Frequencia indica cauda direita longa;")
reg("  e o argumento quantitativo para transformacao logaritmica antes do K-means.")
reg()

# ---------------------------------------------------------------------------
# 4. FIGURA 1 - Distribuicao do valor monetario (original x log)
# ---------------------------------------------------------------------------
reg("-" * 78)
reg("[3] FIGURAS")
reg("-" * 78)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
v = bt.VA_bruto_total.clip(lower=1)
ax[0].hist(v, bins=80, range=(0, v.quantile(.99)), color=COR_PRIMARIA)
ax[0].set_title("(a) Escala original (até o P99)")
ax[0].set_xlabel("VA acumulado bruto (R$)"); ax[0].set_ylabel("Clientes")
ax[0].xaxis.set_major_formatter(FuncFormatter(lambda x, p: milhar(x)))
ax[1].hist(np.log10(v), bins=80, color=COR_OURO)
ax[1].set_title("(b) Escala logarítmica (log$_{10}$)")
ax[1].set_xlabel("log$_{10}$ do VA acumulado bruto"); ax[1].set_ylabel("Clientes")
fig.suptitle("Distribuição do valor monetário por cliente", y=1.02, fontsize=11)
salvar(fig, "aed_distribuicao_monetario.png")

# ---------------------------------------------------------------------------
# 5. FIGURA 2 - Boxplot do VA por grupo de outlier (escala log)
# ---------------------------------------------------------------------------
ordem = ["Outlier Inferior", "Normal", "Outlier Superior"]
fig, ax = plt.subplots(figsize=(7, 3.2))
dados = [bt.loc[bt.flag_outlier == g, "VA_bruto_total"].clip(lower=1) for g in ordem]
_rotulos = [f"{g}\n(n={milhar((bt.flag_outlier==g).sum())})" for g in ordem]
# matplotlib >= 3.9 renomeou 'labels' para 'tick_labels' em boxplot()
_mpl_v = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])
_kw_rot = "tick_labels" if _mpl_v >= (3, 9) else "labels"
bp = ax.boxplot(dados, vert=False, patch_artist=True, widths=.6,
                **{_kw_rot: _rotulos},
                flierprops=dict(marker=".", markersize=2, alpha=.25,
                                markerfacecolor=COR_CINZA, markeredgecolor="none"),
                medianprops=dict(color=COR_PRIMARIA, linewidth=1.6))
for p, g in zip(bp["boxes"], ordem):
    p.set_facecolor(PALETA_GRUPOS[g]); p.set_alpha(.85); p.set_edgecolor(COR_PRIMARIA)
ax.set_xscale("log")
ax.set_xlabel("VA acumulado bruto (R$, escala logarítmica)")
ax.set_title("Valor acumulado bruto por grupo de classificação de outlier")
salvar(fig, "aed_boxplot_grupos.png")

# ---------------------------------------------------------------------------
# 6. FIGURA 3 - Frequencia e recencia
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
vc = bt.faixa_freq.value_counts().reindex(["1", "2", "3", "4-5", "6-10", "11+"])
ax[0].bar(vc.index.astype(str), vc.values, color=COR_PRIMARIA, width=.65)
for i, v_ in enumerate(vc.values):
    ax[0].text(i, v_, f" {v_/len(bt)*100:.0f}%", ha="center", va="bottom", fontsize=8)
ax[0].set_title("(a) Frequência de compra em 2024")
ax[0].set_xlabel("Pedidos no ano"); ax[0].set_ylabel("Clientes")
ax[0].yaxis.set_major_formatter(FuncFormatter(milhar))
ax[0].set_ylim(0, vc.max() * 1.15)

ax[1].hist(bt.recencia, bins=53, color=COR_OURO)
ax[1].axvline(bt.recencia.median(), color=COR_BORDO, linestyle="--", linewidth=1.3,
              label=f"mediana = {bt.recencia.median():.0f} dias")
ax[1].set_title("(b) Recência em 31/12/2024")
ax[1].set_xlabel("Dias desde o último pedido"); ax[1].set_ylabel("Clientes")
ax[1].yaxis.set_major_formatter(FuncFormatter(milhar))
ax[1].legend(fontsize=8)
fig.suptitle("Dimensões transacionais: frequência e recência", y=1.02, fontsize=11)
salvar(fig, "aed_frequencia_recencia.png")

# ---------------------------------------------------------------------------
# 7. FIGURA 4 - Curva de concentracao de receita (Lorenz / Pareto)
# ---------------------------------------------------------------------------
va_ord = np.sort(bt.VA_bruto_total.values)[::-1]
acum = np.cumsum(va_ord) / va_ord.sum() * 100
pct_cli = np.arange(1, len(va_ord) + 1) / len(va_ord) * 100
marcos = {}
for alvo in [1, 5, 10, 20, 50]:
    idx = int(len(va_ord) * alvo / 100) - 1
    marcos[alvo] = acum[idx]

fig, ax = plt.subplots(figsize=(6.6, 3.6))
ax.plot(pct_cli, acum, color=COR_PRIMARIA, linewidth=1.8)
ax.plot([0, 100], [0, 100], color=COR_CINZA, linestyle=":", linewidth=1,
        label="distribuição igualitária")
for alvo in [1, 10, 20]:
    ax.plot([alvo, alvo], [0, marcos[alvo]], color=COR_OURO, linestyle="--", linewidth=1)
    ax.plot(alvo, marcos[alvo], "o", color=COR_OURO, markersize=5)
    ax.annotate(f"{alvo}% dos clientes\n= {marcos[alvo]:.0f}% da receita",
                (alvo, marcos[alvo]), textcoords="offset points", xytext=(12, -14),
                fontsize=8, color=COR_PRIMARIA)
ax.set_xlabel("Clientes acumulados (%, ordenados do maior para o menor VA)")
ax.set_ylabel("Receita acumulada (%)")
ax.set_title("Concentração da receita na base de clientes")
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.legend(fontsize=8, loc="lower right")
salvar(fig, "aed_concentracao_receita.png")

reg()
reg("  Concentracao da receita (base_total):")
for alvo, val in marcos.items():
    reg(f"    top {alvo:>2}% dos clientes -> {val:.1f}% da receita")
_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy>=2 renomeou trapz
gini = 1 - 2 * _trapz(np.sort(bt.VA_bruto_total.values).cumsum() /
                      bt.VA_bruto_total.sum(), dx=1 / len(bt))
reg(f"    indice de Gini da receita: {gini:.3f}")
reg()

# ---------------------------------------------------------------------------
# 8. FIGURA 5 - Canais
# ---------------------------------------------------------------------------
can = bt.groupby("canais_cliente").agg(
    clientes=("churn", "size"), receita=("VA_bruto_total", "sum"),
    churn=("churn", "mean"), va_mediano=("VA_bruto_total", "median"),
    freq_mediana=("Frequencia_total", "median")).sort_values("receita", ascending=False)
can["pct_clientes"] = can.clientes / can.clientes.sum() * 100
can["pct_receita"] = can.receita / can.receita.sum() * 100

ROTULO_CANAL = {"Multicanal": "Multicanal",
                "Exclusivo Varejo": "Exclusivo\nvarejo",
                "Exclusivo Ecommerce": "Exclusivo\ne-commerce"}
rot_can = [ROTULO_CANAL.get(c, c) for c in can.index]

fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
x = np.arange(len(can)); larg = .38
ax[0].bar(x - larg/2, can.pct_clientes, larg, label="% dos clientes", color=COR_CINZA)
ax[0].bar(x + larg/2, can.pct_receita, larg, label="% da receita", color=COR_OURO)
ax[0].set_xticks(x); ax[0].set_xticklabels(rot_can, fontsize=8)
ax[0].set_ylabel("%"); ax[0].legend(fontsize=8, loc="upper right")
ax[0].set_ylim(0, max(can.pct_receita.max(), can.pct_clientes.max()) * 1.28)
ax[0].set_title("(a) Participação por canal")
for i, (c_, r_) in enumerate(zip(can.pct_clientes, can.pct_receita)):
    ax[0].text(i - larg/2, c_, f"{c_:.0f}", ha="center", va="bottom", fontsize=7)
    ax[0].text(i + larg/2, r_, f"{r_:.0f}", ha="center", va="bottom", fontsize=7)

ax[1].bar(x, can.churn * 100, color=COR_BORDO, width=.55)
ax[1].axhline(bt.churn.mean() * 100, color=COR_PRIMARIA, linestyle="--", linewidth=1.1,
              label=f"média da base ({bt.churn.mean()*100:.1f}%)")
ax[1].set_ylabel("Taxa de churn (%)"); ax[1].set_title("(b) Churn por canal")
ax[1].set_xticks(x)
ax[1].set_xticklabels(rot_can, fontsize=8)
ax[1].set_ylim(0, 78); ax[1].legend(fontsize=8, loc="upper left")
for i, v_ in enumerate(can.churn * 100):
    ax[1].text(i, v_, f"{v_:.1f}%", ha="center", va="bottom", fontsize=8)
fig.suptitle("Comportamento por canal de compra", y=1.03, fontsize=11)
salvar(fig, "aed_canais.png")

reg("  Perfil por canal (base_total):")
reg(can.assign(receita=can.receita.round(0)).round(2).to_string())
reg()

# ---------------------------------------------------------------------------
# 9. FIGURA 6 - Idade e distribuicao geografica
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
idv = bt.loc[bt.idade_valida == 1, "idade_fim_2024"]
ax[0].hist(idv, bins=range(10, 102, 2), color=COR_PRIMARIA)
ax[0].axvline(idv.median(), color=COR_OURO, linestyle="--", linewidth=1.4,
              label=f"mediana = {idv.median():.0f} anos")
ax[0].set_title(f"(a) Idade ({(bt.idade_valida==1).mean()*100:.1f}% da base)")
ax[0].set_xlabel("Idade ao fim de 2024"); ax[0].set_ylabel("Clientes")
ax[0].yaxis.set_major_formatter(FuncFormatter(milhar)); ax[0].legend(fontsize=8)

top = bt.estado.value_counts().head(10).sort_values()
ax[1].barh(top.index, top.values, color=COR_OURO)
for i, v_ in enumerate(top.values):
    ax[1].text(v_, i, f" {v_/len(bt)*100:.1f}%", va="center", fontsize=7.5)
ax[1].set_title("(b) Dez estados com mais clientes")
ax[1].set_xlabel("Clientes")
ax[1].xaxis.set_major_formatter(FuncFormatter(milhar))
ax[1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5))
ax[1].set_xlim(0, top.max() * 1.20)
fig.suptitle("Perfil demográfico e distribuição geográfica", y=1.03, fontsize=11)
salvar(fig, "aed_perfil_geografia.png")

reg(f"  Idade: mediana {idv.median():.0f} | media {idv.mean():.1f} | "
    f"P25 {idv.quantile(.25):.0f} | P75 {idv.quantile(.75):.0f}")
reg(f"  Concentracao geografica: top 3 estados = "
    f"{bt.estado.value_counts().head(3).sum()/len(bt)*100:.1f}% dos clientes")
reg(f"  Top 5: {', '.join(f'{k} ({v/len(bt)*100:.1f}%)' for k,v in bt.estado.value_counts().head(5).items())}")
reg()

# ---------------------------------------------------------------------------
# 10. FIGURA 7 - Painel de churn (o bloco mais importante da Frente B)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(9, 6))
media = bt.churn.mean() * 100

def painel_churn(ax, coluna, titulo, xlabel, ordem=None, cor=COR_BORDO):
    g = bt.groupby(coluna, observed=True).churn.agg(["mean", "size"])
    if ordem is not None:
        g = g.reindex(ordem)
    g = g.dropna()
    ax.bar(g.index.astype(str), g["mean"] * 100, color=cor, width=.6)
    ax.axhline(media, color=COR_PRIMARIA, linestyle="--", linewidth=1.1)
    ax.set_title(titulo, fontsize=9.5)
    ax.set_xlabel(xlabel, fontsize=8.5); ax.set_ylabel("Churn (%)", fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=7.5)
    for i, (v_, n_) in enumerate(zip(g["mean"] * 100, g["size"])):
        ax.text(i, v_, f"{v_:.0f}%", ha="center", va="bottom", fontsize=7.5)
    return g

g1 = painel_churn(axes[0, 0], "faixa_recencia", "(a) Por recência",
                  "Dias desde o último pedido",
                  ["0-30", "31-90", "91-180", "181-270", "271-365"])
g2 = painel_churn(axes[0, 1], "faixa_freq", "(b) Por frequência",
                  "Pedidos em 2024", ["1", "2", "3", "4-5", "6-10", "11+"], COR_PRIMARIA)
g3 = painel_churn(axes[1, 0], "faixa_va", "(c) Por valor monetário (quintis)",
                  "Quintil de VA acumulado bruto",
                  ["Q1 (menor)", "Q2", "Q3", "Q4", "Q5 (maior)"], COR_OURO)
# A base de origem grava "NOVO RETRONANTE" (erro de digitacao na query).
# O rotulo e corrigido APENAS para exibicao; o dado permanece intacto.
ROTULO_STATUS = {
    "NOVO": "Novo", "NOVO RETRONANTE": "Novo Retornante",
    "REATIVADO": "Reativado", "REATIVADO RETORNANTE": "Reativado Retornante",
    "RETIDO": "Retido",
}
bt["status_exibicao"] = bt.status_cliente_ano.map(ROTULO_STATUS).fillna(bt.status_cliente_ano)
g4 = painel_churn(axes[1, 1], "status_exibicao", "(d) Por status do cliente",
                  "Status em 2024",
                  ["Novo", "Novo Retornante", "Reativado", "Reativado Retornante", "Retido"],
                  cor=COR_CINZA)
axes[1, 1].tick_params(axis="x", rotation=25, labelsize=6.5)
fig.suptitle(f"Taxa de churn por dimensão (linha tracejada = média da base, {media:.1f}%)",
             y=1.01, fontsize=11)
fig.tight_layout()
salvar(fig, "aed_churn_painel.png")

reg("  Churn por recencia:");   reg((g1["mean"] * 100).round(1).to_string())
reg("  Churn por frequencia:"); reg((g2["mean"] * 100).round(1).to_string())
reg("  Churn por quintil VA:"); reg((g3["mean"] * 100).round(1).to_string())
reg("  Churn por status:");     reg((g4["mean"] * 100).round(1).to_string())
reg()
reg("  Churn por grupo de outlier:")
reg((bt.groupby("flag_outlier").churn.agg(["size", "mean"])
       .assign(mean=lambda d: (d["mean"] * 100).round(1))).to_string())
reg()

# ---------------------------------------------------------------------------
# 11. FIGURA 8 - Correlacao de Spearman
# ---------------------------------------------------------------------------
COLS_CORR = ["VA_bruto_total", "Frequencia_total", "TM_total", "recencia",
             "Qtd_bruta_total", "tempo_vida_2024", "idade_fim_2024", "churn"]
ROT = ["VA acumulado", "Frequência", "Ticket médio", "Recência",
       "Quantidade", "Tempo de vida", "Idade", "Churn"]
corr = bt[COLS_CORR].corr(method="spearman")

fig, ax = plt.subplots(figsize=(6.2, 5.2))
im = ax.imshow(corr, cmap="RdYlBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(ROT))); ax.set_xticklabels(ROT, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(ROT))); ax.set_yticklabels(ROT, fontsize=8)
for i in range(len(ROT)):
    for j in range(len(ROT)):
        v_ = corr.iloc[i, j]
        ax.text(j, i, f"{v_:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if abs(v_) > .55 else COR_PRIMARIA)
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=.8, label="Correlação de Spearman")
ax.set_title("Correlação entre variáveis (Spearman)")
salvar(fig, "aed_correlacao.png")

reg("  Matriz de correlacao de Spearman:")
reg(corr.round(2).to_string())
reg()

# --- Verificacao da identidade algebrica VA = Frequencia x TM
prod = bt.Frequencia_total * bt.TM_total
erro_rel = ((prod - bt.VA_bruto_total).abs() / bt.VA_bruto_total.replace(0, np.nan))
reg("  IDENTIDADE ALGEBRICA VA = Frequencia x TM:")
reg(f"    erro relativo mediano: {erro_rel.median():.2e}")
reg(f"    clientes com erro < 0,1%: {(erro_rel < .001).mean()*100:.2f}%")
reg(f"    correlacao Spearman VA x (Freq x TM): "
    f"{bt.VA_bruto_total.corr(prod, method='spearman'):.4f}")
reg("    -> as tres variaveis NAO sao independentes; usar as tres no K-means")
reg("       equivale a ponderar a dimensao monetaria duas vezes.")
reg()

# ---------------------------------------------------------------------------
# 12. FIGURA 9 - Sazonalidade
# ---------------------------------------------------------------------------
MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
         "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
fig, ax = plt.subplots(figsize=(7.6, 3.3))
p = bt.mes_1o_pedido.value_counts().reindex(range(1, 13), fill_value=0)
u = bt.mes_ultimo_pedido.value_counts().reindex(range(1, 13), fill_value=0)
x = np.arange(12); larg = .38
ax.bar(x - larg/2, p.values, larg, label="Primeiro pedido do ano", color=COR_PRIMARIA)
ax.bar(x + larg/2, u.values, larg, label="Último pedido do ano", color=COR_OURO)
ax.set_xticks(x); ax.set_xticklabels(MESES)
ax.set_ylabel("Clientes"); ax.yaxis.set_major_formatter(FuncFormatter(milhar))
ax.legend(fontsize=8)
ax.set_title("Sazonalidade: distribuição mensal do primeiro e do último pedido (2024)")
salvar(fig, "aed_sazonalidade.png")

reg("  Sazonalidade - clientes por mes do primeiro pedido:")
reg(pd.Series(p.values, index=MESES).to_string())
reg()

# ---------------------------------------------------------------------------
# 13. Information Value (poder discriminante preliminar para a Frente B)
# ---------------------------------------------------------------------------
reg("-" * 78)
reg("[4] INFORMATION VALUE -- MATERIAL DA ETAPA 5B, nao da AED")
reg("-" * 78)
reg("  O IV e uma tecnica de SELECAO/TRATAMENTO de variaveis, nao de")
reg("  exploracao. Fica calculado aqui por conveniencia, mas o texto")
reg("  correspondente pertence a Etapa 5B (engenharia de variaveis).")
reg()

def iv(df, coluna, alvo="churn"):
    """IV = sum( (%nao-churn - %churn) * ln(%nao-churn / %churn) )."""
    t = df.groupby(coluna, observed=True)[alvo].agg(["size", "sum"])
    t["bom"] = t["size"] - t["sum"]              # churn = 0
    t["mau"] = t["sum"]                          # churn = 1
    t["p_bom"] = t.bom / t.bom.sum()
    t["p_mau"] = t.mau / t.mau.sum()
    t = t[(t.p_bom > 0) & (t.p_mau > 0)]
    t["woe"] = np.log(t.p_bom / t.p_mau)
    return ((t.p_bom - t.p_mau) * t.woe).sum()

CANDIDATAS = {
    "Recência (faixas)": "faixa_recencia",
    "Frequência (faixas)": "faixa_freq",
    "Valor monetário (quintis)": "faixa_va",
    "Status do cliente": "status_cliente_ano",
    "Canais do cliente": "canais_cliente",
    "Canal favorito": "canal_favorito",
    "Grupo de outlier": "flag_outlier",
    "Faixa de idade": "faixa_idade",
    "Recompra em 90 dias": "Recompra_90_dias",
    "Recompra em 365 dias": "Recompra_365_dias",
}
ivs = pd.Series({k: iv(bt, v) for k, v in CANDIDATAS.items()}).sort_values(ascending=False)

def forca(v):
    if v < .02:  return "sem poder"
    if v < .1:   return "fraco"
    if v < .3:   return "médio"
    if v < .5:   return "forte"
    return "muito forte (verificar redundancia)"

reg("  Referencia (Siddiqi): <0,02 sem poder | 0,02-0,1 fraco |")
reg("  0,1-0,3 medio | 0,3-0,5 forte | >0,5 suspeito de redundancia")
reg()
for k, v in ivs.items():
    reg(f"    {k:<28} IV = {v:.3f}   ({forca(v)})")
reg()

fig, ax = plt.subplots(figsize=(7, 3.6))
cores = [COR_BORDO if v > .5 else COR_OURO if v > .3 else COR_PRIMARIA for v in ivs.values]
ax.barh(range(len(ivs)), ivs.values[::-1], color=cores[::-1])
ax.set_yticks(range(len(ivs))); ax.set_yticklabels(ivs.index[::-1], fontsize=8)
for lim, rot in [(.1, "médio"), (.3, "forte"), (.5, "redundância?")]:
    ax.axvline(lim, color=COR_CINZA, linestyle=":", linewidth=1)
    ax.text(lim, -1.05, rot, fontsize=7, ha="center", va="top", color=COR_CINZA)
ax.set_ylim(-.8, len(ivs) - .2)
ax.set_xlabel("Information Value (IV)")
ax.set_title("Poder discriminante preliminar das variáveis candidatas (churn)")
salvar(fig, "aed_information_value.png")

# ---------------------------------------------------------------------------
# 14. Balanceamento do alvo
# ---------------------------------------------------------------------------
reg("-" * 78)
reg("[5] BALANCEAMENTO DA VARIAVEL-ALVO")
reg("-" * 78)
for nome, df in [("base_total", bt), ("base_sem_outliers", bs)]:
    n1 = int(df.churn.sum()); n0 = len(df) - n1
    reg(f"  {nome:<20} churn=1: {milhar(n1)} ({n1/len(df)*100:.1f}%) | "
        f"churn=0: {milhar(n0)} ({n0/len(df)*100:.1f}%) | razao {n1/n0:.2f}:1")
reg()
reg("  Alvo praticamente balanceado -> dispensa reamostragem (SMOTE/undersampling).")
reg()

# ---------------------------------------------------------------------------
# 15. Perfil por grupo de outlier (tabela do texto)
# ---------------------------------------------------------------------------
reg("-" * 78)
reg("[6] PERFIL POR GRUPO DE OUTLIER (base_total)")
reg("-" * 78)
perfil = bt.groupby("flag_outlier").agg(
    n=("churn", "size"), churn=("churn", "mean"),
    va_mediano=("VA_bruto_total", "median"), freq_mediana=("Frequencia_total", "median"),
    tm_mediano=("TM_total", "median"), recencia_mediana=("recencia", "median"),
    multicanal=("canais_cliente", lambda s: (s == "Multicanal").mean()),
    receita=("VA_bruto_total", "sum"))
perfil["pct_clientes"] = perfil.n / perfil.n.sum() * 100
perfil["pct_receita"] = perfil.receita / perfil.receita.sum() * 100
perfil["churn"] = perfil.churn * 100
perfil["multicanal"] = perfil.multicanal * 100
reg(perfil.round(2).to_string())
reg()

# ---------------------------------------------------------------------------
# 16. Gravacao do arquivo de conferencia
# ---------------------------------------------------------------------------
caminho_txt = os.path.join(PASTA, "numeros_etapa4.txt")
with open(caminho_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(_saida))
print(f"\n>> Numeros consolidados em: {caminho_txt}")
print(f">> Figuras geradas em: {FIGS}")
