"""
Etapa 7A - Caracterizacao e comparacao dos clusters (Frente A) do TCC.

Usa os rotulos do K-means (Etapa 6A.1) e do DBSCAN (Etapa 6A.2) e as
bases originais para:
  1. Perfil de cada cluster do K-means na ESCALA ORIGINAL (medianas de
     recencia, frequencia e ticket; VA; receita; churn; canais; status; idade).
  2. Traducao do centroide: z -> ln -> valor (media geometrica), ao lado da
     mediana e da media aritmetica, para mostrar a diferenca entre elas.
  3. Comparacoes: K-means x DBSCAN; base total x base sem outliers;
     clientes de topo (Outlier Superior) dentro do cluster de recorrentes.
Base principal: base_total. Base sem outliers: comparacao de robustez.

Uso: python etapa7a.py [pasta]
Saidas: numeros_etapa7a.txt, Figuras/etapa7a_perfil.png
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

COR_PRIMARIA, COR_OURO, COR_BORDO, COR_CINZA, COR_OSSO = (
    "#1C1C1C", "#B8964F", "#6B2737", "#9A9A9A", "#DCD6CC")
CORES_CL = {1: COR_BORDO, 2: COR_OURO, 3: COR_PRIMARIA}
plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.labelsize": 9, "axes.edgecolor": COR_CINZA,
    "axes.linewidth": 0.8, "axes.grid": True, "grid.color": COR_OSSO,
    "grid.linewidth": 0.6, "grid.alpha": 0.7, "axes.axisbelow": True,
    "legend.frameon": False, "xtick.color": COR_PRIMARIA, "ytick.color": COR_PRIMARIA,
    "text.color": COR_PRIMARIA, "axes.labelcolor": COR_PRIMARIA,
})
PASTA = sys.argv[1] if len(sys.argv) > 1 else "."
DATA_CORTE = pd.Timestamp("2024-12-31")
_saida = []
def reg(t=""):
    print(t); _saida.append(str(t))

def carregar(b):
    df = pd.read_csv(os.path.join(PASTA, f"{b}.csv"), parse_dates=["data_ultimo_pedido_2024"])
    df["recencia"] = (DATA_CORTE - df.data_ultimo_pedido_2024).dt.days
    df["km"] = pd.read_csv(os.path.join(PASTA, f"rotulos_kmeans_{b}.csv")).cluster_kmeans.values
    df["db"] = pd.read_csv(os.path.join(PASTA, f"rotulos_dbscan_{b}.csv")).cluster_dbscan.values
    df["status"] = df.status_cliente_ano.replace({"NOVO RETRONANTE": "NOVO RETORNANTE"})
    return df

def perfil(df, nome):
    reg("=" * 78); reg(f"PERFIL DOS CLUSTERS - {nome} ({len(df):,} clientes)".replace(",", ".")); reg("=" * 78)
    rec_tot = df.VA_bruto_total.sum()
    linhas = []
    for c in sorted(df.km.unique()):
        d = df[df.km == c]
        canais = d.canais_cliente.value_counts(normalize=True) * 100
        st = d.status.value_counts(normalize=True) * 100
        r = {"cluster": f"C{c}", "n": len(d), "pct_clientes": len(d) / len(df) * 100,
             "rec_med": d.recencia.median(), "F_med": d.Frequencia_total.median(),
             "TM_med": d.TM_total.median(), "VA_med": d.VA_bruto_total.median(),
             "pct_receita": d.VA_bruto_total.sum() / rec_tot * 100,
             "churn": d.churn.mean() * 100,
             "multicanal": canais.get("Multicanal", 0), "varejo": canais.get("Exclusivo Varejo", 0),
             "ecom": canais.get("Exclusivo Ecommerce", 0),
             "idade_med": d.loc[d.idade_valida == 1, "idade_fim_2024"].median(),
             "pct_F1": (d.Frequencia_total == 1).mean() * 100}
        linhas.append(r)
        reg(f"\nC{c}: n={r['n']:,} ({r['pct_clientes']:.1f}%) | receita {r['pct_receita']:.1f}% | churn {r['churn']:.1f}%".replace(",", "."))
        reg(f"  medianas: recencia {r['rec_med']:.0f} d | frequencia {r['F_med']:.0f} | ticket R$ {r['TM_med']:.2f} | VA R$ {r['VA_med']:.2f}")
        reg(f"  quartis: recencia {d.recencia.quantile(.25):.0f}-{d.recencia.quantile(.75):.0f} | "
            f"freq {d.Frequencia_total.quantile(.25):.0f}-{d.Frequencia_total.quantile(.75):.0f} | "
            f"ticket {d.TM_total.quantile(.25):.0f}-{d.TM_total.quantile(.75):.0f}")
        reg(f"  % com 1 pedido: {r['pct_F1']:.1f}% | canais: multicanal {r['multicanal']:.1f}% "
            f"varejo {r['varejo']:.1f}% ecom {r['ecom']:.1f}% | idade mediana {r['idade_med']:.0f}")
        reg("  status: " + ", ".join(f"{k} {v:.1f}%" for k, v in st.items()))
        reg(f"  outliers: {d.flag_outlier.value_counts().to_dict()}")
    return pd.DataFrame(linhas)

def traducao(df, b):
    M = pd.read_csv(os.path.join(PASTA, f"matriz_5a_{b}.csv"))
    mu_r, sd_r = df.recencia.mean(), df.recencia.std(ddof=0)
    lf, lt = np.log(df.Frequencia_total), np.log(df.TM_total)
    reg(f"\nTRADUCAO DO CENTROIDE ({b}): z -> escala ln -> valor original")
    reg(f"  parametros: recencia mu={mu_r:.2f} sd={sd_r:.2f} | lnF mu={lf.mean():.4f} sd={lf.std(ddof=0):.4f} | lnTM mu={lt.mean():.4f} sd={lt.std(ddof=0):.4f}")
    out = []
    for c in sorted(df.km.unique()):
        m = (df.km == c).values
        z = M.loc[m, ["z_recencia", "z_ln_frequencia", "z_ln_ticket"]].mean().values
        rec = z[0] * sd_r + mu_r
        lnF = z[1] * lf.std(ddof=0) + lf.mean(); F = np.exp(lnF)
        lnT = z[2] * lt.std(ddof=0) + lt.mean(); T = np.exp(lnT)
        d = df[m]
        reg(f"  C{c}: z=({z[0]:+.2f}, {z[1]:+.2f}, {z[2]:+.2f})")
        reg(f"      recencia: centroide {rec:.0f} d | mediana {d.recencia.median():.0f} | media {d.recencia.mean():.0f}")
        reg(f"      frequencia: ln={lnF:.3f} -> exp={F:.2f} (media geom.) | mediana {d.Frequencia_total.median():.0f} | media {d.Frequencia_total.mean():.2f}")
        reg(f"      ticket: ln={lnT:.3f} -> exp=R$ {T:.2f} (media geom.) | mediana R$ {d.TM_total.median():.2f} | media R$ {d.TM_total.mean():.2f}")
        out.append((c, z, rec, lnF, F, lnT, T))
    return out

bt, bs = carregar("base_total"), carregar("base_sem_outliers")
pt = perfil(bt, "BASE TOTAL")
ps = perfil(bs, "BASE SEM OUTLIERS")
tt = traducao(bt, "base_total")
traducao(bs, "base_sem_outliers")

# ---- Comparacao K-means x DBSCAN (base total) ----
reg("\n" + "=" * 78); reg("K-MEANS x DBSCAN (base total)"); reg("=" * 78)
F = bt.Frequencia_total
grande = bt.db.value_counts().drop(-1, errors="ignore").index[:4]
nomes = {}
for c in grande:
    f = F[bt.db == c]
    nomes[c] = f"{int(f.min())} pedido(s)" if f.min() == f.max() else f"{int(f.min())}+ pedidos"
bt["db_nome"] = bt.db.map(nomes).fillna("demais clusters")
bt.loc[bt.db == -1, "db_nome"] = "ruido"
ct = pd.crosstab(bt.km.map(lambda c: f"C{c}"), bt.db_nome, margins=True)
reg(ct.to_string())
reg("\n% de cada cluster K-means por camada DBSCAN:")
reg((pd.crosstab(bt.km.map(lambda c: f"C{c}"), bt.db_nome, normalize="index") * 100).round(1).to_string())

# ---- Base total x base sem outliers ----
reg("\n" + "=" * 78); reg("BASE TOTAL x BASE SEM OUTLIERS (clientes comuns)"); reg("=" * 78)
m = bt[["ClienteID", "km"]].merge(bs[["ClienteID", "km"]], on="ClienteID", suffixes=("_t", "_s"))
ct2 = pd.crosstab(m.km_t, m.km_s)
reg(ct2.to_string())
mapa = ct2.idxmax(axis=1).to_dict()
conc = (m.km_s == m.km_t.map(mapa)).mean() * 100
reg(f"correspondencia (cluster total -> sem outliers): {mapa} | concordancia: {conc:.1f}% de {len(m):,} clientes".replace(",", "."))

# ---- Clientes de topo dentro do C3 ----
reg("\n" + "=" * 78); reg("OUTLIERS SUPERIORES DENTRO DO CLUSTER DE RECORRENTES (base total)"); reg("=" * 78)
c_rec = pt.loc[pt.F_med.idxmax(), "cluster"]; c_rec_n = int(c_rec[1:])
d = bt[bt.km == c_rec_n]
reg(f"cluster de recorrentes: {c_rec} | outliers superiores fora dele: {((bt.flag_outlier=='Outlier Superior') & (bt.km != c_rec_n)).sum()}")
rec_c = d.VA_bruto_total.sum()
for g, dd in [("Outlier Superior", d[d.flag_outlier == "Outlier Superior"]), ("Demais", d[d.flag_outlier != "Outlier Superior"])]:
    can = dd.canais_cliente.value_counts(normalize=True) * 100
    reg(f"  {g}: n={len(dd):,} ({len(dd)/len(d)*100:.1f}% do cluster) | receita do cluster {dd.VA_bruto_total.sum()/rec_c*100:.1f}% | "
        f"receita total {dd.VA_bruto_total.sum()/bt.VA_bruto_total.sum()*100:.1f}%".replace(",", "."))
    reg(f"     medianas: VA R$ {dd.VA_bruto_total.median():.2f} | freq {dd.Frequencia_total.median():.0f} | ticket R$ {dd.TM_total.median():.2f} | "
        f"recencia {dd.recencia.median():.0f} d | churn {dd.churn.mean()*100:.1f}% | multicanal {can.get('Multicanal',0):.1f}%")

# ---- Figura: perfil ----
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
x = np.arange(len(pt)); w = .38
ax[0].bar(x - w/2, pt.pct_clientes, w, color=COR_CINZA, label="% dos clientes")
ax[0].bar(x + w/2, pt.pct_receita, w, color=COR_OURO, label="% da receita")
for i in x:
    ax[0].text(i - w/2, pt.pct_clientes[i] + 1, f"{pt.pct_clientes[i]:.0f}%", ha="center", fontsize=7.5)
    ax[0].text(i + w/2, pt.pct_receita[i] + 1, f"{pt.pct_receita[i]:.0f}%", ha="center", fontsize=7.5)
ax[0].set_title("(a) Clientes e receita"); ax[0].legend(fontsize=7.5, loc="upper left"); ax[0].set_ylim(0, 80)
ax[1].bar(x, pt.churn, .55, color=[CORES_CL[int(c[1:])] for c in pt.cluster])
ax[1].axhline(bt.churn.mean() * 100, color=COR_CINZA, ls="--", lw=1)
ax[1].text(2.45, bt.churn.mean() * 100 + 1.5, f"média {bt.churn.mean()*100:.1f}%".replace(".", ","), fontsize=7.5, ha="right")
for i in x:
    ax[1].text(i, pt.churn[i] + 1, f"{pt.churn[i]:.1f}%".replace(".", ","), ha="center", fontsize=7.5)
ax[1].set_title("(b) Taxa de churn em 2025"); ax[1].set_ylim(0, 80)
bot = np.zeros(len(pt))
for col, cor, lab in [("multicanal", COR_PRIMARIA, "Multicanal"), ("varejo", COR_CINZA, "Exclusivo loja física"), ("ecom", COR_OSSO, "Exclusivo e-commerce")]:
    ax[2].bar(x, pt[col], .55, bottom=bot, color=cor, label=lab); bot += pt[col].values
ax[2].set_title("(c) Composição por canal"); ax[2].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
for a in ax:
    a.set_xticks(x); a.set_xticklabels(pt.cluster); a.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}%"))
fig.tight_layout(); fig.savefig(os.path.join(PASTA, "Figuras", "etapa7a_perfil.png"), facecolor="white"); plt.close(fig)
reg("\n[figura] etapa7a_perfil.png")
open(os.path.join(PASTA, "numeros_etapa7a.txt"), "w", encoding="utf-8").write("\n".join(_saida) + "\n")
