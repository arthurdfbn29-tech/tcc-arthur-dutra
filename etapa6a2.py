"""
Etapa 6A.2 - Aplicacao do DBSCAN (Frente A) do TCC.

Le as matrizes padronizadas da Etapa 5A e, para cada base analitica:
  1. Grafico da distancia ao k-esimo vizinho (k-dist) com MinPts = 4,
     conforme Ester et al. (1996); epsilon escolhido visualmente no
     primeiro "vale" do grafico ordenado.
  2. DBSCAN no espaco completo (3 variaveis) com (eps, MinPts) escolhidos,
     mais grade de sensibilidade.
  3. Analise complementar: DBSCAN dentro de cada camada de frequencia
     (1, 2, 3 e 4+ pedidos), para verificar se ha grupos alem da
     estratificacao imposta pela natureza discreta da frequencia.

Biblioteca: scikit-learn (DBSCAN, NearestNeighbors). min_samples do
scikit-learn conta o proprio ponto, como a definicao de vizinhanca-eps do
Capitulo 2; por isso o k-dist usa o 4o vizinho contando o proprio ponto.

Uso:  python etapa6a2.py [pasta]
Saidas: numeros_etapa6a2.txt, Figuras/etapa6a2_*.png,
        rotulos_dbscan_<base>.csv (GITIGNORED)
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

COR_PRIMARIA, COR_OURO, COR_BORDO, COR_CINZA, COR_OSSO = (
    "#1C1C1C", "#B8964F", "#6B2737", "#9A9A9A", "#DCD6CC")
plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.labelsize": 9, "axes.edgecolor": COR_CINZA,
    "axes.linewidth": 0.8, "axes.grid": True, "grid.color": COR_OSSO,
    "grid.linewidth": 0.6, "grid.alpha": 0.7, "axes.axisbelow": True,
    "legend.frameon": False, "xtick.color": COR_PRIMARIA, "ytick.color": COR_PRIMARIA,
    "text.color": COR_PRIMARIA, "axes.labelcolor": COR_PRIMARIA,
})
virg = FuncFormatter(lambda v, p: f"{v:.2f}".replace(".", ","))
milhar = FuncFormatter(lambda v, p: f"{v:,.0f}".replace(",", "."))

PASTA = sys.argv[1] if len(sys.argv) > 1 else "."
BASES = ["base_total", "base_sem_outliers"]
NOME_BASE = {"base_total": "Base total", "base_sem_outliers": "Base sem outliers"}
COLS = ["z_recencia", "z_ln_frequencia", "z_ln_ticket"]
MINPTS = 4
# Epsilon lido no primeiro "vale" do grafico k-dist (ver figura etapa6a2_kdist.png)
EPS = {"base_total": 0.30, "base_sem_outliers": 0.30}
GRADE_EPS, GRADE_MP = [0.10, 0.20, 0.30, 0.40], [4, 16]
# Camadas de frequencia e epsilon de cada camada (vale do k-dist da camada)
CAMADAS = [("1 pedido", 1, 1, 0.10), ("2 pedidos", 2, 2, 0.11),
           ("3 pedidos", 3, 3, 0.12), ("4 ou mais", 4, 10**6, 0.25)]
SEED = 42
# Silhueta descritiva (exata, sem o ruido): calculo lento (~1 min/base), feito
# com "python etapa6a2.py . --silhueta" e guardado em cache. Usada apenas para
# comparacao com o K-means, nunca para escolher parametros (ver Capitulo 2).
CACHE = os.path.join(PASTA, "etapa6a2_cache.json")
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

_saida = []
def reg(t=""):
    print(t); _saida.append(str(t))

def kdist(X, k=MINPTS):
    d, _ = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    return np.sort(d[:, k - 1])[::-1]

def resumo(lab):
    v = lab[lab >= 0]
    tam = np.sort(np.bincount(v))[::-1] if len(v) else np.array([0])
    return len(tam) if len(v) else 0, (lab < 0).mean(), tam

reg("=" * 78); reg("ETAPA 6A.2 - DBSCAN"); reg("=" * 78)
reg(f"scikit-learn {sklearn.__version__} | MinPts = {MINPTS} (Ester et al., 1996)")
dados = {}
for b in BASES:
    df = pd.read_csv(os.path.join(PASTA, f"{b}.csv"),
                     usecols=["Frequencia_total", "VA_bruto_total", "flag_outlier"])
    M = pd.read_csv(os.path.join(PASTA, f"matriz_5a_{b}.csv"))
    X = M[COLS].values
    kd = kdist(X)
    eps = EPS[b]
    lab = DBSCAN(eps=eps, min_samples=MINPTS).fit(X).labels_
    dados[b] = (df, M, X, kd, lab)
    nc, ruido, tam = resumo(lab)
    reg(); reg(f"--- {b} ({len(X):,} clientes) ---".replace(",", "."))
    reg(f"k-dist (4o vizinho): mediana={np.median(kd):.3f} p95={np.quantile(kd,.95):.3f} "
        f"p99={np.quantile(kd,.99):.3f} max={kd.max():.3f}")
    reg(f"eps escolhido = {eps} | clusters = {nc} | ruido = {ruido*100:.2f}% ({(lab<0).sum()})")
    reg(f"  5 maiores clusters: {tam[:5].tolist()} | clusters com < 100 clientes: {(tam<100).sum()}")
    # composicao por frequencia dos maiores clusters
    F = df.Frequencia_total.clip(upper=4).values
    ordem = pd.Series(lab[lab >= 0]).value_counts().index[:4]
    ordem_4 = max(ordem, key=lambda c: F[lab == c].mean())
    for c in ordem:
        m = lab == c
        comp = pd.Series(F[m]).value_counts(normalize=True).sort_index() * 100
        reg(f"  cluster {c}: n={m.sum()} | frequencia (4 = 4+): " +
            ", ".join(f"{int(i)}: {v:.1f}%" for i, v in comp.items()))
    reg(f"  ruido por grupo: {df.flag_outlier[lab < 0].value_counts().to_dict()}")
    top = (df.VA_bruto_total >= df.VA_bruto_total.quantile(.99)).values
    reg(f"  top 1% de VA: {top.sum()} | no ruido: {(lab[top] < 0).sum()} | "
        f"no cluster de 4+ pedidos: {(lab[top] == ordem_4).sum()}")
    sd = np.log(df.Frequencia_total).std(ddof=0)
    gaps = [(np.log(f + 1) - np.log(f)) / sd for f in range(1, 6)]
    reg("  distancia entre niveis consecutivos de frequencia (z): " +
        ", ".join(f"{f}->{f+1}: {g:.3f}" for f, g in zip(range(1, 6), gaps)))
    chave = f"{b}_eps{eps}"
    if chave not in cache and "--silhueta" in sys.argv and nc > 1:
        mm = lab >= 0
        cache[chave] = float(silhouette_score(X[mm], lab[mm]))
        json.dump(cache, open(CACHE, "w"), indent=1)
    if chave in cache:
        reg(f"  silhueta media (exata, sem ruido): {cache[chave]:.4f}")
    reg("  Sensibilidade (clusters | ruido% | % do maior cluster):")
    for mp in GRADE_MP:
        linha = []
        for e in GRADE_EPS:
            l2 = DBSCAN(eps=e, min_samples=mp).fit(X).labels_
            n2, r2, t2 = resumo(l2)
            linha.append(f"eps={e:.2f}: {n2:>4} | {r2*100:4.1f}% | {t2[0]/len(X)*100:4.1f}%")
        reg(f"    MinPts={mp:>2}  " + "  ;  ".join(linha))
    pd.DataFrame({"ClienteID": M.ClienteID, "cluster_dbscan": lab}).to_csv(
        os.path.join(PASTA, f"rotulos_dbscan_{b}.csv"), index=False)
    reg(f"  [rotulos] rotulos_dbscan_{b}.csv")

    # ---- Analise por camada de frequencia --------------------------------
    reg("  Analise por camada de frequencia (MinPts = 4):")
    for nome, fmin, fmax, e in CAMADAS:
        m = ((df.Frequencia_total >= fmin) & (df.Frequencia_total <= fmax)).values
        cols = ["z_recencia", "z_ln_ticket"] if fmin == fmax else COLS
        Xc = M.loc[m, cols].values
        for ee in [e * .5, e, e * 1.5]:
            lc = DBSCAN(eps=ee, min_samples=MINPTS).fit(Xc).labels_
            n3, r3, t3 = resumo(lc)
            marca = " <- eps do vale" if ee == e else ""
            reg(f"    {nome:<10} n={m.sum():>6} eps={ee:.3f}: clusters={n3:>4} "
                f"ruido={r3*100:4.1f}% maior={t3[0]/len(Xc)*100:5.1f}% "
                f"seguintes={t3[1:4].tolist()}{marca}")

# ---------------------------------------------------------------------------
# Figura 1: grafico k-dist das duas bases
fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
for b, cor in [("base_total", COR_PRIMARIA), ("base_sem_outliers", COR_OURO)]:
    kd = dados[b][3]
    ax[0].plot(np.arange(len(kd)), kd, color=cor, lw=1.3, label=NOME_BASE[b])
    ax[1].plot(np.arange(3000), kd[:3000], color=cor, lw=1.3, label=NOME_BASE[b])
ax[1].axhline(0.30, color=COR_BORDO, ls="--", lw=1.1, label="ε = 0,30")
ax[0].set_title("(a) Todos os clientes"); ax[1].set_title("(b) Detalhe: 3.000 maiores valores")
ax[1].set_ylim(0, 1)
for a in ax:
    a.set_xlabel("Clientes ordenados pela distância (decrescente)")
    a.set_ylabel("Distância ao 4º vizinho (z)")
    a.xaxis.set_major_formatter(milhar); a.yaxis.set_major_formatter(virg); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(PASTA, "Figuras", "etapa6a2_kdist.png"), facecolor="white"); plt.close(fig)
reg(); reg("[figura] etapa6a2_kdist.png")

# Figura 2: DBSCAN global (base total) - projecao ln(freq) x ln(ticket) e recencia x ln(freq)
df, M, X, kd, lab = dados["base_total"]
amostra = np.random.RandomState(SEED).choice(len(M), 15000, replace=False)
ordem = pd.Series(lab[lab >= 0]).value_counts().index[:4]
paleta = [COR_CINZA, COR_OURO, COR_PRIMARIA, "#4F6D7A"]
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
for a, (cx, cy, lx, ly) in zip(ax, [("z_ln_frequencia", "z_ln_ticket", "ln(frequência)", "ln(ticket médio)"),
                                    ("z_recencia", "z_ln_frequencia", "Recência", "ln(frequência)")]):
    la = lab[amostra]
    outros = ~np.isin(la, list(ordem)) & (la >= 0)
    for c, cor in zip(ordem, paleta):
        m = la == c
        n_c = (lab == c).sum()
        a.scatter(M[cx].values[amostra][m], M[cy].values[amostra][m], s=2, alpha=.3, color=cor,
                  rasterized=True, label=f"Cluster de {n_c:,} clientes".replace(",", "."))
    a.scatter(M[cx].values[amostra][outros], M[cy].values[amostra][outros], s=6, color="#4F6D7A", marker="^", label="Demais clusters")
    m = la < 0
    a.scatter(M[cx].values[amostra][m], M[cy].values[amostra][m], s=10, color=COR_BORDO, marker="x", lw=.8, label="Ruído")
    a.set_xlabel(lx + " (z)"); a.set_ylabel(ly + " (z)")
ax[0].set_title("(a) ln(frequência) × ln(ticket médio)"); ax[1].set_title("(b) Recência × ln(frequência)")
ax[1].legend(fontsize=7, loc="upper right", markerscale=3)
fig.tight_layout(); fig.savefig(os.path.join(PASTA, "Figuras", "etapa6a2_dbscan_global.png"), facecolor="white"); plt.close(fig)
reg("[figura] etapa6a2_dbscan_global.png")

# Figura 3: DBSCAN dentro de cada camada (base total): recencia x ln(ticket)
fig, ax = plt.subplots(1, 4, figsize=(12, 3.3), sharey=True)
for a, (nome, fmin, fmax, e) in zip(ax, CAMADAS):
    m = ((df.Frequencia_total >= fmin) & (df.Frequencia_total <= fmax)).values
    cols = ["z_recencia", "z_ln_ticket"] if fmin == fmax else COLS
    Xc = M.loc[m, cols].values
    lc = DBSCAN(eps=e, min_samples=MINPTS).fit(Xc).labels_
    principal = np.bincount(lc[lc >= 0]).argmax()
    rec, tk = M.loc[m, "z_recencia"].values, M.loc[m, "z_ln_ticket"].values
    idx = np.random.RandomState(SEED).choice(m.sum(), min(6000, m.sum()), replace=False)
    g = lc[idx]
    a.scatter(rec[idx][g == principal], tk[idx][g == principal], s=2, alpha=.3, color=COR_CINZA, rasterized=True)
    sec = (g >= 0) & (g != principal)
    a.scatter(rec[idx][sec], tk[idx][sec], s=8, color=COR_OURO, marker="^")
    a.scatter(rec[idx][g < 0], tk[idx][g < 0], s=10, color=COR_BORDO, marker="x", lw=.8)
    pct = (lc == principal).mean() * 100
    a.set_title(f"{nome}\nmaior cluster: {pct:.1f}%".replace(".", ","), fontsize=9)
    a.set_xlabel("Recência (z)"); a.set_ylim(-6.5, 4.5)
ax[0].set_ylabel("ln(ticket médio) (z)")
h = [plt.Line2D([], [], marker="o", ls="", color=COR_CINZA, label="Cluster principal"),
     plt.Line2D([], [], marker="^", ls="", color=COR_OURO, label="Clusters menores"),
     plt.Line2D([], [], marker="x", ls="", color=COR_BORDO, label="Ruído")]
ax[3].legend(handles=h, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(PASTA, "Figuras", "etapa6a2_dbscan_camadas.png"), facecolor="white"); plt.close(fig)
reg("[figura] etapa6a2_dbscan_camadas.png")
open(os.path.join(PASTA, "numeros_etapa6a2.txt"), "w", encoding="utf-8").write("\n".join(_saida) + "\n")
