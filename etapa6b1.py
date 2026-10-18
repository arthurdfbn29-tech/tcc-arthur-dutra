"""
Etapa 6B.1 - Treinamento da Regressao Logistica (Frente B) do TCC.

Para cada base: ajusta a regressao logistica por MAXIMA VEROSSIMILHANCA
(Newton-Raphson, sem regularizacao) no conjunto de TREINO preparado na 5B,
com statsmodels (Seabold e Perktold, 2010). Entrega:
  - coeficientes, erros-padrao, p-valores, razoes de chance (OR) e IC 95%;
  - OR em unidades interpretaveis (mesmo coeficiente, reexpresso):
      recencia: +30 dias | frequencia e ticket: ao DOBRAR | idade: +10 anos;
  - Teste de Razao de Verossimilhanca global e por variavel (blocos);
  - limiar q = proporcao de churn no treino (Wichitaksorn et al., 2023);
  - previa de desempenho (treino e teste); probabilidades do teste p/ a 7B.

Uso: python etapa6b1.py [pasta]
Saidas: numeros_etapa6b1.txt, Figuras/etapa6b1_razoes_chance.png,
        6b1_probabilidades_<base>.csv (GITIGNORED)
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels, statsmodels.api as sm
from scipy import stats
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
DATA_CORTE = pd.Timestamp("2024-12-31")
_saida = []
def reg(t=""):
    print(t); _saida.append(str(t))

ROTULO = {"z_recencia": "Recência", "z_ln_frequencia": "Frequência", "z_ln_ticket": "Ticket médio",
          "z_idade_ajustada": "Idade", "idade_valida": "Idade válida (sim)",
          "canais_cliente=Exclusivo Ecommerce": "Exclusivo e-commerce",
          "canais_cliente=Exclusivo Varejo": "Exclusivo loja física",
          "status=NOVO": "Status: novo", "status=NOVO RETORNANTE": "Status: novo retornante",
          "status=REATIVADO": "Status: reativado", "status=REATIVADO RETORNANTE": "Status: reativado retornante",
          "recompra90=Janela incompleta": "Recompra 90d: janela incompleta",
          "recompra90=Nao recomprou": "Recompra 90d: não recomprou"}
BLOCOS = {"Recência": ["z_recencia"], "Frequência": ["z_ln_frequencia"], "Ticket médio": ["z_ln_ticket"],
          "Idade (valor e indicador)": ["z_idade_ajustada", "idade_valida"],
          "Canais do cliente": ["canais_cliente=Exclusivo Ecommerce", "canais_cliente=Exclusivo Varejo"],
          "Status do cliente": ["status=NOVO", "status=NOVO RETORNANTE", "status=REATIVADO", "status=REATIVADO RETORNANTE"],
          "Recompra em 90 dias": ["recompra90=Janela incompleta", "recompra90=Nao recomprou"]}

def desvios_treino(b):
    """Recalcula os desvios-padrao do treino (os mesmos da 5B) para reexpressar as OR."""
    tr = pd.read_csv(os.path.join(PASTA, f"5b_{b}_treino_arvore.csv"))
    return {"z_recencia": tr.recencia.std(ddof=0), "z_ln_frequencia": np.log(tr.Frequencia_total).std(ddof=0),
            "z_ln_ticket": np.log(tr.TM_total).std(ddof=0), "z_idade_ajustada": tr.idade_ajustada.std(ddof=0)}

def ajustar(X, y):
    return sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(method="newton", maxiter=100, disp=0)

reg("=" * 78); reg("ETAPA 6B.1 - REGRESSAO LOGISTICA"); reg("=" * 78)
reg(f"statsmodels {statsmodels.__version__} | Logit por maxima verossimilhanca (Newton-Raphson), sem regularizacao")
res_fig = {}
for b in ["base_total", "base_sem_outliers"]:
    tr = pd.read_csv(os.path.join(PASTA, f"5b_{b}_treino_logistica.csv"))
    te = pd.read_csv(os.path.join(PASTA, f"5b_{b}_teste_logistica.csv"))
    cols = [c for c in tr.columns if c not in ("ClienteID", "churn")]
    Xtr, ytr, Xte, yte = tr[cols].astype(float), tr.churn, te[cols].astype(float), te.churn
    m = ajustar(Xtr, ytr)
    sd = desvios_treino(b)
    reg(); reg("-" * 78); reg(f"{b}: treino n={len(tr):,} | convergiu={m.mle_retvals['converged']} | iteracoes={m.mle_retvals['iterations']}".replace(",", ".")); reg("-" * 78)
    ci = m.conf_int()
    tab = pd.DataFrame({"beta": m.params, "ep": m.bse, "z": m.tvalues, "p": m.pvalues,
                        "OR": np.exp(m.params), "OR_inf": np.exp(ci[0]), "OR_sup": np.exp(ci[1])})
    reg(tab.round(4).to_string())
    # OR em unidades interpretaveis
    fator = {"z_recencia": 30 / sd["z_recencia"], "z_ln_frequencia": np.log(2) / sd["z_ln_frequencia"],
             "z_ln_ticket": np.log(2) / sd["z_ln_ticket"], "z_idade_ajustada": 10 / sd["z_idade_ajustada"]}
    unid = {"z_recencia": "+30 dias", "z_ln_frequencia": "ao dobrar", "z_ln_ticket": "ao dobrar", "z_idade_ajustada": "+10 anos"}
    reg("OR em unidades interpretaveis (mesmo coeficiente, reexpresso):")
    interp = {}
    for v, f in fator.items():
        o, lo, hi = np.exp(m.params[v] * f), np.exp(ci.loc[v, 0] * f), np.exp(ci.loc[v, 1] * f)
        interp[v] = (o, lo, hi)
        reg(f"  {ROTULO[v]:<14} {unid[v]:<10} OR={o:.3f} [{lo:.3f}; {hi:.3f}]  (por desvio-padrao: {np.exp(m.params[v]):.3f})")
    # TRV global
    G = 2 * (m.llf - m.llnull); gl = int(m.df_model)
    reg(f"TRV global: -2lnL nulo={-2*m.llnull:,.1f} | -2lnL completo={-2*m.llf:,.1f} | G={G:,.1f} | gl={gl} | p={stats.chi2.sf(G, gl):.2e}".replace(",", "."))
    reg(f"  pseudo-R2 de McFadden (referencia) = {1 - m.llf/m.llnull:.4f}")
    # TRV por bloco
    reg("TRV por variavel (modelo completo x modelo sem a variavel):")
    trv = []
    for nome, bl in BLOCOS.items():
        mr = ajustar(Xtr.drop(columns=bl), ytr)
        Gb = 2 * (m.llf - mr.llf)
        trv.append((nome, len(bl), Gb, stats.chi2.sf(Gb, len(bl))))
    trv = sorted(trv, key=lambda x: -x[2])
    for nome, gl_b, Gb, p in trv:
        reg(f"  {nome:<26} gl={gl_b}  G={Gb:>9,.1f}  p={p:.2e}  ({Gb/G*100:.1f}% do G global)".replace(",", "."))
    # limiar e previa
    q = ytr.mean()
    reg(f"Limiar q = proporcao de churn no treino = {q:.4f}")
    for nome, X, y in [("treino", Xtr, ytr), ("teste", Xte, yte)]:
        p_ = m.predict(sm.add_constant(X, has_constant="add"))
        yh = (p_ > q).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
        reg(f"  [{nome}] acuracia={accuracy_score(y, yh):.4f} precisao={precision_score(y, yh):.4f} "
            f"recall={recall_score(y, yh):.4f} F1={f1_score(y, yh):.4f} AUC={roc_auc_score(y, p_):.4f} "
            f"| VP={tp} FP={fp} FN={fn} VN={tn}")
        if nome == "teste":
            pd.DataFrame({"ClienteID": te.ClienteID, "churn": y, "prob_logistica": p_.values}).to_csv(
                os.path.join(PASTA, f"6b1_probabilidades_{b}.csv"), index=False)
    res_fig[b] = (tab, interp)

# ---------------------------------------------------------------------------
# Figura: OR com IC 95% (base total), em unidades interpretaveis
tab, interp = res_fig["base_total"]
itens = []
for v, (o, lo, hi) in interp.items():
    u = {"z_recencia": "+30 dias", "z_ln_frequencia": "dobrar", "z_ln_ticket": "dobrar", "z_idade_ajustada": "+10 anos"}[v]
    itens.append((f"{ROTULO[v]} ({u})", o, lo, hi, "num"))
for v in ROTULO:
    if v in interp: continue
    itens.append((ROTULO[v], tab.loc[v, "OR"], tab.loc[v, "OR_inf"], tab.loc[v, "OR_sup"], "cat"))
itens = sorted(itens, key=lambda x: x[1])
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for i, (n, o, lo, hi, t) in enumerate(itens):
    cor = COR_BORDO if o > 1 else COR_OURO
    ax.plot([lo, hi], [i, i], color=cor, lw=1.6)
    ax.scatter([o], [i], color=cor, s=28, zorder=3, marker="o" if t == "num" else "s")
    ax.text(hi * 1.08, i, f"{o:.2f}".replace(".", ","), va="center", fontsize=7.5)
ax.axvline(1, color=COR_CINZA, ls="--", lw=1)
ax.set_xscale("log")
ax.set_yticks(range(len(itens))); ax.set_yticklabels([x[0] for x in itens], fontsize=8)
ax.set_xlabel("Razão de chances de churn (escala logarítmica)")
ticks = [0.1, 0.2, 0.5, 1, 2, 5]
ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}".replace(".", ",") for t in ticks])
ax.set_xlim(min(x[2] for x in itens) * 0.8, max(x[3] for x in itens) * 1.6)
fig.savefig(os.path.join(PASTA, "Figuras", "etapa6b1_razoes_chance.png"), facecolor="white"); plt.close(fig)
reg(); reg("[figura] etapa6b1_razoes_chance.png")
open(os.path.join(PASTA, "numeros_etapa6b1.txt"), "w", encoding="utf-8").write("\n".join(_saida) + "\n")
