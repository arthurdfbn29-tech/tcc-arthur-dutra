"""
Etapa 6B.2 - Treinamento da Arvore de Decisao (Frente B) do TCC.

CART (Breiman et al., 1984): criterio Gini, divisoes binarias, poda por
custo-complexidade. O alfa da poda e escolhido por validacao cruzada
estratificada com 10 partes, SOMENTE no conjunto de treino, por duas regras:
  (a) alfa de menor erro medio de validacao cruzada;
  (b) regra de 1 erro-padrao: arvore mais simples com erro <= minimo + 1 EP.
Limiar q = proporcao de churn no treino (mesmo criterio da logistica).

Uso: python etapa6b2.py [pasta] [base]   (base: base_total | base_sem_outliers)
     sem base: gera numeros/figuras a partir do cache das duas bases.
Saidas: numeros_etapa6b2.txt, Figuras/etapa6b2_*.png,
        6b2_probabilidades_<base>.csv (GITIGNORED), etapa6b2_cache.json
"""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

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
BASE = sys.argv[2] if len(sys.argv) > 2 else None
SEED, V, N_ALFAS = 42, 10, 40
CACHE = os.path.join(PASTA, "etapa6b2_cache.json")
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
NOMES = {"recencia": "Recência (dias)", "Frequencia_total": "Frequência (pedidos)", "TM_total": "Ticket médio (R$)",
         "idade_ajustada": "Idade", "idade_valida": "Idade válida",
         "canais_cliente=Exclusivo Ecommerce": "Exclusivo e-commerce", "canais_cliente=Exclusivo Varejo": "Exclusivo loja física",
         "canais_cliente=Multicanal": "Multicanal", "status=NOVO": "Status: novo", "status=NOVO RETORNANTE": "Status: novo retornante",
         "status=REATIVADO": "Status: reativado", "status=REATIVADO RETORNANTE": "Status: reativado retornante",
         "status=RETIDO": "Status: retido", "recompra90=Janela incompleta": "Recompra 90d: janela incompleta",
         "recompra90=Nao recomprou": "Recompra 90d: não recomprou", "recompra90=Recomprou": "Recompra 90d: recomprou"}

def dados(b):
    tr = pd.read_csv(os.path.join(PASTA, f"5b_{b}_treino_arvore.csv"))
    te = pd.read_csv(os.path.join(PASTA, f"5b_{b}_teste_arvore.csv"))
    cols = [c for c in tr.columns if c not in ("ClienteID", "churn")]
    return tr, te, cols

def arvore(a):
    return DecisionTreeClassifier(criterion="gini", ccp_alpha=a, random_state=SEED)

def metricas(m, X, y, q):
    p = m.predict_proba(X)[:, 1]; yh = (p > q).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
    return {"acuracia": accuracy_score(y, yh), "precisao": precision_score(y, yh), "recall": recall_score(y, yh),
            "f1": f1_score(y, yh), "auc": roc_auc_score(y, p), "VP": int(tp), "FP": int(fp), "FN": int(fn), "VN": int(tn)}, p

def regras(m, cols):
    t = m.tree_; out = []
    def rec(n, cond):
        if t.children_left[n] == -1:
            v = t.value[n][0]; tot = t.n_node_samples[n]
            frac = v[1] / v.sum()
            out.append((" E ".join(cond) if cond else "(raiz)", int(tot), float(frac)))
            return
        f, thr = cols[t.feature[n]], t.threshold[n]
        nome = NOMES.get(f, f)
        if f in ("recencia", "Frequencia_total", "TM_total", "idade_ajustada"):
            a, b = f"{nome} <= {thr:.2f}", f"{nome} > {thr:.2f}"
        else:
            a, b = f"{nome} = não", f"{nome} = sim"
        rec(t.children_left[n], cond + [a]); rec(t.children_right[n], cond + [b])
    rec(0, []); return out

# ---------------------------------------------------------------------------
if BASE:
    tr, te, cols = dados(BASE)
    X, y = tr[cols].values, tr.churn.values
    caminho = arvore(0).cost_complexity_pruning_path(X, y)
    alfas_all = np.unique(caminho.ccp_alphas)
    alfas_all = alfas_all[alfas_all > 0]
    # grade: valores de alfa log-espacados entre 10^-6 e 10^-2 (cobre desde a
    # arvore quase completa ate arvores de poucas folhas)
    alfas = np.logspace(-6, -2, N_ALFAS)
    cv = StratifiedKFold(n_splits=V, shuffle=True, random_state=SEED)
    res = []
    for a in alfas:
        acc = cross_val_score(arvore(a), X, y, cv=cv, scoring="accuracy", n_jobs=2)
        err = 1 - acc
        folhas = arvore(a).fit(X, y).get_n_leaves()
        res.append({"alfa": float(a), "erro": float(err.mean()), "ep": float(err.std(ddof=1) / np.sqrt(V)), "folhas": int(folhas)})
    cache[BASE] = {"n_alfas_poda": int(len(alfas_all)), "folhas_completa": int(arvore(0).fit(X, y).get_n_leaves()), "cv": res}
    json.dump(cache, open(CACHE, "w"), indent=1)
    print(BASE, "ok", len(res), "alfas")
    sys.exit(0)

# ---------------------------------------------------------------------------
_saida = []
def reg(t=""):
    print(t); _saida.append(str(t))
reg("=" * 78); reg("ETAPA 6B.2 - ARVORE DE DECISAO (CART)"); reg("=" * 78)
reg(f"scikit-learn {sklearn.__version__} | Gini | poda custo-complexidade | VC estratificada {V} partes no treino | {N_ALFAS} alfas avaliados")
finais = {}
for b in ["base_total", "base_sem_outliers"]:
    tr, te, cols = dados(b)
    X, y, Xt, yt = tr[cols].values, tr.churn.values, te[cols].values, te.churn.values
    c = cache[b]; cvr = pd.DataFrame(c["cv"])
    i_min = cvr.erro.idxmin(); lim = cvr.erro[i_min] + cvr.ep[i_min]
    i_1se = cvr[cvr.erro <= lim].alfa.idxmax()
    q = y.mean()
    reg(); reg("-" * 78); reg(f"{b}: arvore completa = {c['folhas_completa']} folhas | {c['n_alfas_poda']} valores de alfa na sequencia de poda"); reg("-" * 78)
    reg(cvr.assign(erro=cvr.erro.round(5), ep=cvr.ep.round(5)).to_string(index=False))
    reg(f"(a) minimo: alfa={cvr.alfa[i_min]:.3e} erro={cvr.erro[i_min]:.4f} (EP {cvr.ep[i_min]:.4f}) folhas={cvr.folhas[i_min]}")
    reg(f"(b) 1-EP : alfa={cvr.alfa[i_1se]:.3e} erro={cvr.erro[i_1se]:.4f} (limite {lim:.4f}) folhas={cvr.folhas[i_1se]}")
    reg(f"limiar q = {q:.4f}")
    finais[b] = {}
    for rotulo, i in [("minimo", i_min), ("1EP", i_1se)]:
        m = arvore(cvr.alfa[i]).fit(X, y)
        mt, _ = metricas(m, X, y, q); me, p = metricas(m, Xt, yt, q)
        reg(f"  [{rotulo}] folhas={m.get_n_leaves()} profundidade={m.get_depth()}")
        for nome, mm in [("treino", mt), ("teste", me)]:
            reg(f"     {nome}: acuracia={mm['acuracia']:.4f} precisao={mm['precisao']:.4f} recall={mm['recall']:.4f} "
                f"F1={mm['f1']:.4f} AUC={mm['auc']:.4f} | VP={mm['VP']} FP={mm['FP']} FN={mm['FN']} VN={mm['VN']}")
        # sensibilidade do limiar: folhas com taxa entre q e 0.5
        t = m.tree_; folhas = t.children_left == -1
        fr = t.value[folhas][:, 0, 1] / t.value[folhas][:, 0].sum(axis=1)
        reg(f"     folhas com taxa de churn entre q e 0,5: {((fr > q) & (fr <= .5)).sum()} "
            f"({t.n_node_samples[folhas][(fr > q) & (fr <= .5)].sum()} clientes do treino)")
        imp = pd.Series(m.feature_importances_, index=[NOMES[c_] for c_ in cols]).sort_values(ascending=False)
        reg("     importancia (reducao de Gini): " + "; ".join(f"{k} {v:.3f}" for k, v in imp.items() if v > 0.001))
        finais[b][rotulo] = (m, imp, me, cvr.folhas[i])
        pd.DataFrame({"ClienteID": te.ClienteID, "churn": yt, f"prob_arvore": p}).to_csv(
            os.path.join(PASTA, f"6b2_probabilidades_{b}_{rotulo}.csv"), index=False)
    # regras da arvore 1-EP
    m = finais[b]["1EP"][0]
    rr = sorted(regras(m, cols), key=lambda x: -x[2])
    grandes = [r for r in rr if r[1] >= 0.01 * len(tr)]
    reg(f"  Regras (arvore 1-EP, {len(rr)} folhas; exibidas as {len(grandes)} com >= 1% do treino), por taxa de churn:")
    for cond, n, fr_ in grandes:
        reg(f"     churn {fr_*100:5.1f}% | n={n:>6} ({n/len(tr)*100:4.1f}%) | {cond}")
    reg(f"  cobertura das folhas exibidas: {sum(r[1] for r in grandes)/len(tr)*100:.1f}% do treino")

# ---------------------------------------------------------------------------
# Figura 1: curva de validacao cruzada (base total)
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for b, cor, mk in [("base_total", COR_PRIMARIA, "o"), ("base_sem_outliers", COR_OURO, "s")]:
    cvr = pd.DataFrame(cache[b]["cv"]).sort_values("folhas")
    ax.errorbar(cvr.folhas, cvr.erro, yerr=cvr.ep, color=cor, marker=mk, ms=3, lw=1.1, capsize=2,
                label="Base total" if b == "base_total" else "Base sem outliers")
    for rot, mk2 in [("minimo", "v"), ("1EP", "D")]:
        fo = finais[b][rot][3]; e = cvr.set_index("folhas").erro.get(fo)
        if np.ndim(e): e = e.iloc[0]
        ax.scatter([fo], [e], s=70, facecolors="none", edgecolors=COR_BORDO, linewidths=1.4, marker=mk2, zorder=5)
ax.set_xscale("log"); ax.set_xlabel("Número de folhas da árvore podada (escala logarítmica)")
ax.set_ylabel("Erro de classificação (VC)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:.3f}".replace(".", ",")))
h, l = ax.get_legend_handles_labels()
h += [plt.Line2D([], [], marker="v", ls="", mfc="none", mec=COR_BORDO, label="Erro mínimo"),
      plt.Line2D([], [], marker="D", ls="", mfc="none", mec=COR_BORDO, label="Regra de 1 erro-padrão")]
ax.legend(handles=h, fontsize=7.5)
fig.savefig(os.path.join(PASTA, "Figuras", "etapa6b2_validacao_cruzada.png"), facecolor="white"); plt.close(fig)

# Figura 2: arvore 1-EP (base total), primeiros niveis, rotulos em portugues
import re
m = finais["base_total"]["1EP"][0]; _, _, cols = dados("base_total")
fig, ax = plt.subplots(figsize=(17, 6.6))
anots = plot_tree(m, max_depth=3, feature_names=[NOMES[c_].replace(" (dias)", "").replace(" (pedidos)", "").replace("Ticket médio (R$)", "Ticket médio") for c_ in cols], class_names=["Retido", "Churn"],
                  filled=False, impurity=False, proportion=True, rounded=True, fontsize=7, ax=ax, precision=3)
def fmt(v):
    return f"{v:.1f}".replace(".", ",")
for a in anots:
    t = a.get_text()
    t = t.replace("True", "Sim").replace("False", "Não")
    linhas = []
    for l in t.split("\n"):
        mt = re.match(r"(.+) <= ([0-9.]+)$", l)
        if mt:
            nome, thr = mt.group(1), float(mt.group(2))
            if abs(thr - 0.5) < 1e-9 and nome not in ("Frequência",):
                linhas.append(f"{nome}: não?")
            else:
                linhas.append(f"{nome} <= {fmt(thr)}?")
        elif l.startswith("samples"):
            linhas.append("clientes: " + l.split("=")[1].strip().replace(".", ","))
        elif l.startswith("value"):
            v = re.findall(r"[0-9.]+", l); linhas.append(f"churn: {float(v[1])*100:.1f}%".replace(".", ","))
        elif l.startswith("class"):
            pass
        else:
            linhas.append(l)
    a.set_text("\n".join(linhas))
fig.savefig(os.path.join(PASTA, "Figuras", "etapa6b2_arvore.png"), facecolor="white"); plt.close(fig)

# Figura 3: importancia (base total, 1-EP e minimo)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
for a, rot, tit in [(ax[0], "1EP", "(a) Árvore pela regra de 1 erro-padrão"), (ax[1], "minimo", "(b) Árvore de erro mínimo")]:
    imp = finais["base_total"][rot][1]; imp = imp[imp > 0.001][::-1]
    a.barh(imp.index, imp.values, color=COR_PRIMARIA)
    for i_, v in enumerate(imp.values):
        a.text(v + .005, i_, f"{v:.3f}".replace(".", ","), va="center", fontsize=7)
    a.set_title(tit); a.set_xlabel("Importância (redução do índice Gini)"); a.tick_params(axis="y", labelsize=7.5)
    a.set_xlim(0, imp.max() * 1.2)
fig.tight_layout(); fig.savefig(os.path.join(PASTA, "Figuras", "etapa6b2_importancia.png"), facecolor="white"); plt.close(fig)
reg(); reg("[figuras] etapa6b2_validacao_cruzada.png, etapa6b2_arvore.png, etapa6b2_importancia.png")
open(os.path.join(PASTA, "numeros_etapa6b2.txt"), "w", encoding="utf-8").write("\n".join(_saida) + "\n")
