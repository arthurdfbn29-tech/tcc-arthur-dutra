"""
Etapa 6A.1 - Aplicacao do K-means (Frente A) do TCC.

Le as matrizes padronizadas da Etapa 5A e, para cada base analitica:
  1. Executa o K-means (algoritmo de Lloyd, inicializacao aleatoria) para
     k = 2..10, com 20 inicializacoes por k; guarda a de menor SSE e
     registra quantas inicializacoes chegaram ao mesmo SSE (estabilidade).
  2. Calcula o SSE (Metodo do Cotovelo) e o Coeficiente de Silhueta medio
     EXATO, sobre todos os clientes (sem amostragem).
  3. Escolhe k pela regra definida ANTES de ver os resultados:
       - k com maior silhueta media entre k >= 3 (k = 2 e reportado, mas
         nao elegivel: dois grupos nao constituem segmentacao utilizavel);
       - o cotovelo e usado como conferencia e a divergencia e registrada.
  4. Salva rotulos (CSV, GITIGNORED), tamanhos e centroides do k escolhido.

Biblioteca: scikit-learn (KMeans, silhouette_score).
O calculo exato da silhueta e quadratico em n e leva ~1 min por k na base
total; por isso os resultados sao guardados em cache (etapa6a1_cache.json)
e o script pode ser executado em partes:  python etapa6a1.py . [k_max_por_execucao]

Saidas:
  numeros_etapa6a1.txt, Figuras/etapa6a1_cotovelo_silhueta.png,
  rotulos_kmeans_<base>.csv (GITIGNORED)
"""
import sys, os, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
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

PASTA = sys.argv[1] if len(sys.argv) > 1 else "."
MAX_NOVOS = int(sys.argv[2]) if len(sys.argv) > 2 else 99
BASES = ["base_total", "base_sem_outliers"]
NOME_BASE = {"base_total": "Base total", "base_sem_outliers": "Base sem outliers"}
KS = list(range(2, 11))
N_INIT, SEED, MAX_ITER = 20, 42, 1000
COLS = ["z_recencia", "z_ln_frequencia", "z_ln_ticket"]
CACHE = os.path.join(PASTA, "etapa6a1_cache.json")

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
_saida = []
def reg(t=""):
    print(t); _saida.append(str(t))


def kmeans_k(X, k):
    """20 execucoes independentes com inicializacao aleatoria; retorna a melhor."""
    sementes = np.random.RandomState(SEED + k).randint(0, 2**31 - 1, N_INIT)
    melhor, sses, iters = None, [], []
    for s in sementes:
        km = KMeans(n_clusters=k, init="random", n_init=1, algorithm="lloyd",
                    max_iter=MAX_ITER, tol=0.0, random_state=int(s)).fit(X)
        sses.append(km.inertia_); iters.append(km.n_iter_)
        if melhor is None or km.inertia_ < melhor.inertia_:
            melhor = km
    sses = np.array(sses)
    iguais = int((np.abs(sses - sses.min()) / sses.min() < 1e-6).sum())
    return melhor, sses, iters, iguais


novos = 0
dados = {}
for b in BASES:
    M = pd.read_csv(os.path.join(PASTA, f"matriz_5a_{b}.csv"))
    X = M[COLS].values
    dados[b] = (M, X)
    cache.setdefault(b, {})
    for k in KS:
        if str(k) in cache[b] or novos >= MAX_NOVOS:
            continue
        t0 = time.time()
        km, sses, iters, iguais = kmeans_k(X, k)
        sil = float(silhouette_score(X, km.labels_))
        cache[b][str(k)] = {"sse": float(km.inertia_), "sil": sil,
                            "sse_pior": float(sses.max()), "iguais": iguais,
                            "iter_max": int(max(iters)),
                            "nao_convergiu": int(sum(i >= MAX_ITER for i in iters)),
                            "tamanhos": np.bincount(km.labels_).tolist()}
        json.dump(cache, open(CACHE, "w"), indent=1)
        novos += 1
        print(f"  {b} k={k}: SSE={km.inertia_:.1f} sil={sil:.4f} ({time.time()-t0:.0f}s)")

pendentes = [(b, k) for b in BASES for k in KS if str(k) not in cache[b]]
if pendentes:
    print(f"PENDENTES: {len(pendentes)} -> rode novamente"); sys.exit(0)

# ---------------------------------------------------------------------------
reg("=" * 78); reg("ETAPA 6A.1 - K-MEANS"); reg("=" * 78)
reg(f"scikit-learn {sklearn.__version__} | init=random | n_init={N_INIT} | "
    f"algoritmo de Lloyd | parada: atribuicoes estaveis (tol=0, max_iter={MAX_ITER})")
reg("Silhueta: media exata sobre todos os clientes (sem amostragem)")
escolha = {}
for b in BASES:
    M, X = dados[b]; c = cache[b]
    reg(); reg(f"--- {b} ({len(X):,} clientes) ---".replace(",", "."))
    reg(f"{'k':>3}{'SSE':>12}{'reducao%':>10}{'silhueta':>10}{'inits=melhor':>14}{'iter_max':>9}  tamanhos")
    ant = None
    for k in KS:
        r = c[str(k)]
        red = "" if ant is None else f"{(ant - r['sse']) / ant * 100:.1f}"
        reg(f"{k:>3}{r['sse']:>12.1f}{red:>10}{r['sil']:>10.4f}{r['iguais']:>9}/{N_INIT}"
            f"{r['iter_max']:>9}  {sorted(r['tamanhos'], reverse=True)}")
        ant = r["sse"]
    elegiveis = [k for k in KS if k >= 3]
    k_sil = max(elegiveis, key=lambda k: c[str(k)]["sil"])
    k2 = c["2"]["sil"]
    escolha[b] = k_sil
    reg(f"k escolhido (maior silhueta, k>=3): {k_sil}  (silhueta k=2: {k2:.4f})")

    # Ajuste final do k escolhido e saidas
    km, *_ = kmeans_k(X, k_sil)
    # Ordena clusters por tamanho (C1 = maior) para rotulos estaveis
    ordem = np.argsort(-np.bincount(km.labels_))
    mapa = {old: new + 1 for new, old in enumerate(ordem)}
    rot = np.vectorize(mapa.get)(km.labels_)
    pd.DataFrame({"ClienteID": M.ClienteID, "cluster_kmeans": rot}).to_csv(
        os.path.join(PASTA, f"rotulos_kmeans_{b}.csv"), index=False)
    reg(f"  Clusters (ordenados por tamanho) e centroides no espaco padronizado:")
    for new, old in enumerate(ordem, 1):
        n = (km.labels_ == old).sum(); ce = km.cluster_centers_[old]
        reg(f"    C{new}: n={n:>6} ({n/len(X)*100:5.1f}%)  "
            f"z_rec={ce[0]:+.2f}  z_lnF={ce[1]:+.2f}  z_lnTM={ce[2]:+.2f}")
    reg(f"  [rotulos] rotulos_kmeans_{b}.csv")

# ---------------------------------------------------------------------------
# Analise de sensibilidade: onde caem os clientes do 1% de maior valor
# acumulado (da propria base) quando k aumenta de 3 a 7?
# ---------------------------------------------------------------------------
reg(); reg("SENSIBILIDADE: alocacao do top 1% de valor acumulado, k = 3..7")
for b in BASES:
    M, X = dados[b]
    df = pd.read_csv(os.path.join(PASTA, f"{b}.csv"), usecols=["VA_bruto_total", "Frequencia_total"])
    top = (df.VA_bruto_total >= df.VA_bruto_total.quantile(.99)).values
    reg(f"--- {b} (top 1% = {top.sum()} clientes)")
    for k in range(3, 8):
        km, *_ = kmeans_k(X, k)
        lab = km.labels_
        c_top = np.bincount(lab[top], minlength=k).argmax()
        n_c = (lab == c_top).sum(); n_top_c = (lab[top] == c_top).sum()
        tam = sorted(np.bincount(lab).tolist(), reverse=True)
        freq_med = [int(np.median(df.Frequencia_total.values[lab == j])) for j in range(k)]
        n_f1 = sum(1 for j in range(k) if freq_med[j] == 1)
        reg(f"  k={k}: cluster do top1% n={n_c} | top1% nele={n_top_c} "
            f"({n_top_c/top.sum()*100:.1f}% do top) | peso no cluster={n_top_c/n_c*100:.1f}% "
            f"| clusters com freq. mediana = 1: {n_f1} | tamanhos {tam}")

# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for b, cor, mk in [("base_total", COR_PRIMARIA, "o"), ("base_sem_outliers", COR_OURO, "s")]:
    c = cache[b]
    sse = [c[str(k)]["sse"] / 1000 for k in KS]
    sil = [c[str(k)]["sil"] for k in KS]
    ax[0].plot(KS, sse, marker=mk, color=cor, label=NOME_BASE[b], linewidth=1.4, markersize=4)
    ax[1].plot(KS, sil, marker=mk, color=cor, label=NOME_BASE[b], linewidth=1.4, markersize=4)
    kk = escolha[b]
    ax[1].scatter([kk], [c[str(kk)]["sil"]], s=90, facecolors="none", edgecolors=COR_BORDO, linewidths=1.5, zorder=5)
ax[0].set_title("(a) Método do Cotovelo"); ax[0].set_xlabel("Número de clusters (k)")
ax[0].set_ylabel("SSE (milhares)")
ax[1].set_title("(b) Coeficiente de Silhueta médio"); ax[1].set_xlabel("Número de clusters (k)")
ax[1].set_ylabel("Silhueta média")
for a in ax:
    a.set_xticks(KS); a.legend(fontsize=8)
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:.2f}".replace(".", ",") if a is ax[1] else f"{v:,.0f}".replace(",", ".")))
ax[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:.2f}".replace(".", ",")))
ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:,.0f}".replace(",", ".")))
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "Figuras", "etapa6a1_cotovelo_silhueta.png"), facecolor="white")
plt.close(fig)
reg(); reg("[figura] etapa6a1_cotovelo_silhueta.png")
# ---------------------------------------------------------------------------
# Figura: projecoes 2D do espaco padronizado coloridas pelo cluster (k escolhido)
# Exibe amostra aleatoria de 15.000 clientes por base apenas para legibilidade.
# ---------------------------------------------------------------------------
CORES_CL = {1: COR_BORDO, 2: COR_OURO, 3: COR_PRIMARIA}
EIXOS = [("z_recencia", "z_ln_frequencia", "Recência", "ln(frequência)"),
         ("z_recencia", "z_ln_ticket", "Recência", "ln(ticket médio)"),
         ("z_ln_frequencia", "z_ln_ticket", "ln(frequência)", "ln(ticket médio)")]
fig, ax = plt.subplots(2, 3, figsize=(10, 6.4))
for i, b in enumerate(BASES):
    M, X = dados[b]
    rot = pd.read_csv(os.path.join(PASTA, f"rotulos_kmeans_{b}.csv")).cluster_kmeans.values
    amostra = np.random.RandomState(SEED).choice(len(M), 15000, replace=False)
    for j, (cx, cy, lx, ly) in enumerate(EIXOS):
        a = ax[i, j]
        for c in sorted(CORES_CL):
            m = rot[amostra] == c
            a.scatter(M[cx].values[amostra][m], M[cy].values[amostra][m], s=2, alpha=.25,
                      color=CORES_CL[c], label=f"C{c}", rasterized=True)
        for c in sorted(CORES_CL):
            mm = rot == c
            a.scatter(M[cx].values[mm].mean(), M[cy].values[mm].mean(), s=70, marker="X",
                      color=CORES_CL[c], edgecolors="white", linewidths=.8, zorder=5)
        a.set_xlabel(lx + " (z)"); a.set_ylabel(ly + " (z)")
        if j == 0:
            a.set_title(NOME_BASE[b], loc="left")
    if i == 0:
        h = [plt.Line2D([], [], marker="o", ls="", color=CORES_CL[c], label=f"C{c}") for c in sorted(CORES_CL)]
        ax[0, 2].legend(handles=h, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(PASTA, "Figuras", "etapa6a1_clusters_projecoes.png"), facecolor="white")
plt.close(fig)
reg("[figura] etapa6a1_clusters_projecoes.png")

open(os.path.join(PASTA, "numeros_etapa6a1.txt"), "w", encoding="utf-8").write("\n".join(_saida) + "\n")
