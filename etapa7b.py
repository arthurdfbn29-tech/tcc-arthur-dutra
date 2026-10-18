"""
Etapa 7B - Avaliacao e comparacao dos modelos supervisionados (Frente B).

Usa SOMENTE as probabilidades do conjunto de TESTE salvas nas Etapas 6B.1 e
6B.2 (modelos ja fechados; nada e reajustado). Para cada base:
  1. Metricas (acuracia, precisao, recall, F1, AUC-ROC) e matrizes de confusao,
     com limiar q = proporcao de churn no treino;
  2. Intervalos de confianca de 95% por bootstrap pareado (1.000 reamostras
     com reposicao do conjunto de teste) para as metricas e para a diferenca
     entre os modelos;
  3. Curvas ROC; concordancia entre os modelos;
  4. Cenarios: variacao do limiar e capacidade de acao (top X% de risco);
  5. Recall ponderado pelo valor acumulado de 2024 (lente do luxo).

Uso: python etapa7b.py [pasta]
Saidas: numeros_etapa7b.txt, Figuras/etapa7b_*.png
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

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
virg = lambda d: plt.FuncFormatter(lambda v, p: f"{v:.{d}f}".replace(".", ","))
PASTA = sys.argv[1] if len(sys.argv) > 1 else "."
B, SEED = 1000, 42
_saida = []
def reg(t=""):
    print(t); _saida.append(str(t))

def carregar(b):
    lr = pd.read_csv(os.path.join(PASTA, f"6b1_probabilidades_{b}.csv"))
    ar = pd.read_csv(os.path.join(PASTA, f"6b2_probabilidades_{b}_1EP.csv")).rename(columns={"prob_arvore": "prob_arvore"})
    am = pd.read_csv(os.path.join(PASTA, f"6b2_probabilidades_{b}_minimo.csv")).rename(columns={"prob_arvore": "prob_arvore_min"})
    d = lr.merge(ar[["ClienteID", "prob_arvore"]], on="ClienteID").merge(am[["ClienteID", "prob_arvore_min"]], on="ClienteID")
    va = pd.read_csv(os.path.join(PASTA, f"{b}.csv"), usecols=["ClienteID", "VA_bruto_total"])
    d = d.merge(va, on="ClienteID", how="left")
    q = pd.read_csv(os.path.join(PASTA, f"5b_{b}_treino_arvore.csv"), usecols=["churn"]).churn.mean()
    assert len(d) == len(lr) and d.VA_bruto_total.notna().all()
    return d, q

def metr(y, p, q, w=None):
    yh = (p > q).astype(int)
    out = {"acuracia": accuracy_score(y, yh), "precisao": precision_score(y, yh, zero_division=0),
           "recall": recall_score(y, yh), "f1": f1_score(y, yh), "auc": roc_auc_score(y, p)}
    if w is not None:
        out["recall_valor"] = (w * (y == 1) * (yh == 1)).sum() / (w * (y == 1)).sum()
    return out

MODELOS = [("Regressão logística", "prob_logistica"), ("Árvore (1 erro-padrão)", "prob_arvore"), ("Árvore (erro mínimo)", "prob_arvore_min")]
reg("=" * 78); reg("ETAPA 7B - AVALIACAO E COMPARACAO DOS MODELOS (conjunto de teste)"); reg("=" * 78)
reg(f"bootstrap pareado: {B} reamostras com reposicao, semente {SEED}; IC 95% por percentis")
guarda = {}
for b in ["base_total", "base_sem_outliers"]:
    d, q = carregar(b); y = d.churn.values; w = d.VA_bruto_total.values
    reg(); reg("-" * 78); reg(f"{b}: teste n={len(d):,} | churn {y.mean()*100:.2f}% | q={q:.4f}".replace(",", ".")); reg("-" * 78)
    pont = {nome: metr(y, d[col].values, q, w) for nome, col in MODELOS}
    for nome, col in MODELOS:
        yh = (d[col].values > q).astype(int); tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
        m = pont[nome]
        reg(f"  {nome:<24} acur={m['acuracia']:.4f} prec={m['precisao']:.4f} rec={m['recall']:.4f} F1={m['f1']:.4f} "
            f"AUC={m['auc']:.4f} rec_valor={m['recall_valor']:.4f} | VP={tp} FP={fp} FN={fn} VN={tn}")
    # bootstrap pareado
    rng = np.random.RandomState(SEED); n = len(d)
    bs = {nome: [] for nome, _ in MODELOS}; dif = []; dif_min = []
    for _ in range(B):
        i = rng.randint(0, n, n)
        r = {nome: metr(y[i], d[col].values[i], q, w[i]) for nome, col in MODELOS}
        for nome in bs: bs[nome].append(r[nome])
        dif.append({k: r["Regressão logística"][k] - r["Árvore (1 erro-padrão)"][k] for k in r["Regressão logística"]})
        dif_min.append({k: r["Árvore (erro mínimo)"][k] - r["Árvore (1 erro-padrão)"][k] for k in r["Regressão logística"]})
    reg("  IC 95% (bootstrap):")
    ic = {}
    for nome in bs:
        df_ = pd.DataFrame(bs[nome]); ic[nome] = df_.quantile([.025, .975])
        reg(f"    {nome:<24} " + " | ".join(f"{k} [{df_[k].quantile(.025):.4f}; {df_[k].quantile(.975):.4f}]" for k in df_.columns))
    dd = pd.DataFrame(dif)
    reg("  Diferenca logistica - arvore(1EP): " + " | ".join(
        f"{k} {pont['Regressão logística'][k]-pont['Árvore (1 erro-padrão)'][k]:+.4f} [{dd[k].quantile(.025):+.4f}; {dd[k].quantile(.975):+.4f}]"
        f"{' *' if (dd[k].quantile(.025) > 0 or dd[k].quantile(.975) < 0) else ''}" for k in dd.columns))
    dm = pd.DataFrame(dif_min)
    reg("  Diferenca arvore(min) - arvore(1EP): " + " | ".join(
        f"{k} {pont['Árvore (erro mínimo)'][k]-pont['Árvore (1 erro-padrão)'][k]:+.4f} [{dm[k].quantile(.025):+.4f}; {dm[k].quantile(.975):+.4f}]"
        f"{' *' if (dm[k].quantile(.025) > 0 or dm[k].quantile(.975) < 0) else ''}" for k in dm.columns))
    reg("  (* = intervalo nao inclui zero)")
    # concordancia
    a = (d.prob_logistica.values > q).astype(int); t = (d.prob_arvore.values > q).astype(int)
    reg(f"  Concordancia logistica x arvore: {(a == t).mean()*100:.1f}% | ambos churn {((a==1)&(t==1)).mean()*100:.1f}% "
        f"| ambos retido {((a==0)&(t==0)).mean()*100:.1f}% | so logistica churn {((a==1)&(t==0)).mean()*100:.1f}% | so arvore churn {((a==0)&(t==1)).mean()*100:.1f}%")
    disc = a != t
    reg(f"  Entre os {disc.sum()} discordantes: churn real {y[disc].mean()*100:.1f}% | acerto logistica {(a[disc]==y[disc]).mean()*100:.1f}% | acerto arvore {(t[disc]==y[disc]).mean()*100:.1f}%")
    # valor
    reg(f"  Valor: churners no teste = {(y==1).sum()} | VA dos churners = R$ {w[y==1].sum():,.0f} ({w[y==1].sum()/w.sum()*100:.1f}% do VA do teste)".replace(",", "."))
    for nome, col in MODELOS[:2]:
        yh = d[col].values > q
        perdidos = (y == 1) & ~yh
        reg(f"    {nome}: recall (clientes) {pont[nome]['recall']*100:.1f}% | recall (valor) {pont[nome]['recall_valor']*100:.1f}% "
            f"| VA mediano churners detectados R$ {np.median(w[(y==1)&yh]):,.0f} x nao detectados R$ {np.median(w[perdidos]):,.0f}".replace(",", "."))
    # capacidade de acao
    reg("  Capacidade de acao (clientes de maior risco):")
    cap = {}
    for nome, col in MODELOS[:2]:
        p = d[col].values; ordem = np.argsort(-p, kind="stable"); linhas = []
        for frac in [.1, .2, .3, .4, .5]:
            k = int(round(frac * n)); sel = np.zeros(n, bool); sel[ordem[:k]] = True
            prec = y[sel].mean(); rec = y[sel].sum() / y.sum(); recv = (w * sel * (y == 1)).sum() / (w * (y == 1)).sum()
            linhas.append((frac, prec, rec, recv))
            reg(f"    {nome:<24} top {int(frac*100)}%: precisao {prec*100:.1f}% | churners alcancados {rec*100:.1f}% | valor alcancado {recv*100:.1f}%")
        cap[nome] = linhas
    reg(f"    referencia (selecao aleatoria): precisao = taxa de churn = {y.mean()*100:.1f}%")
    guarda[b] = (d, q, pont, ic, dd, cap)

# ---------------------------------------------------------------------------
d, q, pont, ic, dd, cap = guarda["base_total"]; y = d.churn.values
# Figura 1: ROC
fig, ax = plt.subplots(figsize=(4.8, 4.4))
for (nome, col), cor in zip(MODELOS[:2], [COR_PRIMARIA, COR_OURO]):
    fpr, tpr, _ = roc_curve(y, d[col].values)
    ax.plot(fpr, tpr, color=cor, lw=1.6, label=f"{nome} (AUC = {pont[nome]['auc']:.3f})".replace(".", ","))
    yh = d[col].values > q; tpr_q = (yh & (y == 1)).sum() / (y == 1).sum(); fpr_q = (yh & (y == 0)).sum() / (y == 0).sum()
    ax.scatter([fpr_q], [tpr_q], color=cor, s=30, zorder=5, edgecolors="white")
ax.plot([0, 1], [0, 1], color=COR_CINZA, ls="--", lw=1, label="Classificação aleatória")
ax.set_xlabel("Taxa de falsos positivos"); ax.set_ylabel("Taxa de verdadeiros positivos (recall)")
ax.xaxis.set_major_formatter(virg(1)); ax.yaxis.set_major_formatter(virg(1)); ax.legend(fontsize=7.5, loc="lower right")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.savefig(os.path.join(PASTA, "Figuras", "etapa7b_roc.png"), facecolor="white"); plt.close(fig)

# Figura 2: limiar
qs = np.linspace(.2, .8, 61)
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
for a_, (nome, col) in zip(ax, MODELOS[:2]):
    p = d[col].values
    pr = [precision_score(y, (p > t).astype(int), zero_division=0) for t in qs]
    rc = [recall_score(y, (p > t).astype(int)) for t in qs]
    f1 = [f1_score(y, (p > t).astype(int)) for t in qs]
    a_.plot(qs, pr, color=COR_OURO, lw=1.5, label="Precisão"); a_.plot(qs, rc, color=COR_BORDO, lw=1.5, label="Recall")
    a_.plot(qs, f1, color=COR_PRIMARIA, lw=1.2, ls="--", label="F1")
    a_.axvline(q, color=COR_CINZA, ls=":", lw=1.2); a_.text(q + .01, .05, f"q = {q:.3f}".replace(".", ","), fontsize=7, color=COR_CINZA)
    a_.set_title(nome); a_.set_xlabel("Limiar de decisão"); a_.xaxis.set_major_formatter(virg(1)); a_.yaxis.set_major_formatter(virg(1))
    a_.set_ylim(0, 1.02)
ax[0].set_ylabel("Valor da métrica"); ax[0].legend(fontsize=7.5, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(PASTA, "Figuras", "etapa7b_limiar.png"), facecolor="white"); plt.close(fig)

# Figura 3: capacidade de acao
fig, ax = plt.subplots(figsize=(6.2, 3.6))
x = np.array([10, 20, 30, 40, 50])
for (nome, _), cor, mk in zip(MODELOS[:2], [COR_PRIMARIA, COR_OURO], ["o", "s"]):
    ln = cap[nome]
    ax.plot(x, [l[2]*100 for l in ln], color=cor, marker=mk, lw=1.5, label=f"{nome}: clientes em churn alcançados")
    ax.plot(x, [l[3]*100 for l in ln], color=cor, marker=mk, lw=1.1, ls="--", mfc="white", label=f"{nome}: valor alcançado")
ax.plot(x, x, color=COR_CINZA, ls=":", lw=1, label="Seleção aleatória")
ax.set_xlabel("Clientes selecionados, em ordem decrescente de risco (% do conjunto de teste)")
ax.set_ylabel("% alcançado"); ax.set_xticks(x); ax.legend(fontsize=6.8, loc="upper left"); ax.set_ylim(0, 100)
fig.savefig(os.path.join(PASTA, "Figuras", "etapa7b_capacidade.png"), facecolor="white"); plt.close(fig)
reg(); reg("[figuras] etapa7b_roc.png, etapa7b_limiar.png, etapa7b_capacidade.png")
open(os.path.join(PASTA, "numeros_etapa7b.txt"), "w", encoding="utf-8").write("\n".join(_saida) + "\n")
