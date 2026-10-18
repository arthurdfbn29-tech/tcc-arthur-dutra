"""
Etapa 5B - Definicao de churn e engenharia de variaveis (Frente B) do TCC.

Para cada base analitica:
  1. Alvo: churn = 1 se a cliente ativa em 2024 nao fez nenhuma compra em 2025
     (coluna ja construida no SQL; aqui apenas conferida).
  2. Information Value (IV) das variaveis candidatas (base total), com as
     faixas de Siddiqi (2006). Mesma funcao iv() do aed.py.
  3. Recompra em 90 dias tratada em 3 categorias: recomprou / nao recomprou /
     janela incompleta (1a compra a partir de 03/10/2024 e sem recompra observada).
  4. Divisao treino/teste 70/30 ALEATORIA E ESTRATIFICADA pelo churn
     (semente fixa), com tabela de verificacao do equilibrio.
  5. Apos a divisao: imputacao da idade pela mediana DO TREINO + indicador;
     ln e escore-z (parametros DO TREINO) para a regressao logistica;
     variaveis indicadoras para as categoricas. A arvore usa escala original.

Uso: python etapa5b.py [pasta]
Saidas: numeros_etapa5b.txt, Figuras/etapa5b_information_value.png,
        5b_<base>_<treino|teste>_<logistica|arvore>.csv (GITIGNORED)
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

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
INICIO_TRUNC = DATA_CORTE - pd.Timedelta(days=89)   # 03/10/2024: janela de 90 dias nao cabe em 2024
SEED, TESTE = 42, 0.30
_saida = []
def reg(t=""):
    print(t); _saida.append(str(t))

def carregar(b):
    df = pd.read_csv(os.path.join(PASTA, f"{b}.csv"),
                     parse_dates=["data_1_pedido_2024", "data_ultimo_pedido_2024"])
    df["recencia"] = (DATA_CORTE - df.data_ultimo_pedido_2024).dt.days
    df["status"] = df.status_cliente_ano.replace({"NOVO RETRONANTE": "NOVO RETORNANTE"})
    trunc = df.data_1_pedido_2024 >= INICIO_TRUNC
    df["recompra90"] = np.where(df.Recompra_90_dias == 1, "Recomprou",
                        np.where(trunc, "Janela incompleta", "Nao recomprou"))
    # faixas (mesmas da Etapa 4) para o IV
    df["faixa_freq"] = pd.cut(df.Frequencia_total, [0, 1, 2, 3, 5, 10, np.inf],
                              labels=["1", "2", "3", "4-5", "6-10", "11+"])
    df["faixa_recencia"] = pd.cut(df.recencia, [-1, 30, 90, 180, 270, 366],
                                  labels=["0-30", "31-90", "91-180", "181-270", "271-365"])
    df["faixa_va"] = pd.qcut(df.VA_bruto_total, 5, labels=False)
    df["faixa_tm"] = pd.qcut(df.TM_total, 5, labels=False)
    ok = df.idade_valida == 1
    df["faixa_idade"] = np.where(ok, pd.cut(df.idade_fim_2024, [0, 25, 35, 45, 55, 100],
                        labels=["ate 25", "26-35", "36-45", "46-55", "56+"]).astype(object), "sem informacao")
    df["estado_iv"] = np.where(df.estado_valido == 1, df.estado, "invalido")
    return df, trunc

def iv(df, coluna, alvo="churn"):
    t = df.groupby(coluna, observed=True)[alvo].agg(["size", "sum"])
    t["bom"] = t["size"] - t["sum"]; t["mau"] = t["sum"]
    t["p_bom"] = t.bom / t.bom.sum(); t["p_mau"] = t.mau / t.mau.sum()
    t = t[(t.p_bom > 0) & (t.p_mau > 0)]
    return ((t.p_bom - t.p_mau) * np.log(t.p_bom / t.p_mau)).sum()

def forca(v):
    return ("sem poder" if v < .02 else "fraco" if v < .1 else "medio" if v < .3
            else "forte" if v < .5 else "muito alto")

reg("=" * 78); reg("ETAPA 5B - DEFINICAO DE CHURN E ENGENHARIA DE VARIAVEIS"); reg("=" * 78)
reg(f"scikit-learn {sklearn.__version__} | teste={TESTE} | semente={SEED} | estratificado por churn")

# ---------------------------------------------------------------------------
bt, trunc_t = carregar("base_total")
reg("\n[1] ALVO (base total)")
reg(f"  churn=1: {bt.churn.sum():,} ({bt.churn.mean()*100:.1f}%) | churn=0: {(bt.churn==0).sum():,}".replace(",", "."))

reg("\n[2] RECOMPRA EM 90 DIAS - 3 categorias (base total)")
reg(f"  primeira compra >= {INICIO_TRUNC.date()} (janela truncada): {trunc_t.sum():,} ({trunc_t.mean()*100:.1f}%)".replace(",", "."))
g = bt.groupby("recompra90").churn.agg(["size", "mean"])
for k, r in g.iterrows():
    reg(f"  {k:<18} n={int(r['size']):>6,} ({r['size']/len(bt)*100:4.1f}%) | churn {r['mean']*100:.1f}%".replace(",", "."))
reg(f"  entre os de janela truncada, recompra observada: {(bt.Recompra_90_dias[trunc_t]==1).mean()*100:.1f}% "
    f"| janela completa: {(bt.Recompra_90_dias[~trunc_t]==1).mean()*100:.1f}%")

CAND = {"Frequência (faixas)": "faixa_freq", "Valor acumulado (quintis)": "faixa_va",
        "Status do cliente": "status", "Recompra em 365 dias": "Recompra_365_dias",
        "Canais do cliente": "canais_cliente", "Recompra em 90 dias (3 categorias)": "recompra90",
        "Recompra em 90 dias (original)": "Recompra_90_dias",
        "Recência (faixas)": "faixa_recencia", "Ticket médio (quintis)": "faixa_tm",
        "Grupo de outlier": "flag_outlier", "Faixa de idade": "faixa_idade",
        "Estado": "estado_iv", "Canal favorito": "canal_favorito"}
ivs = pd.Series({k: iv(bt, v) for k, v in CAND.items()}).sort_values(ascending=False)
reg("\n[3] INFORMATION VALUE (base total) - faixas de Siddiqi (2006)")
for k, v in ivs.items():
    reg(f"  {k:<36} IV = {v:.3f}  ({forca(v)})")

# figura IV (sem a versao original da recompra 90, para nao duplicar)
ivf = ivs.drop("Recompra em 90 dias (original)")
fig, ax = plt.subplots(figsize=(7.2, 4.0))
cores = [COR_BORDO if v > .5 else COR_OURO if v > .3 else COR_CINZA if v < .02 else COR_PRIMARIA for v in ivf.values]
ax.barh(range(len(ivf)), ivf.values[::-1], color=cores[::-1])
ax.set_yticks(range(len(ivf))); ax.set_yticklabels(ivf.index[::-1], fontsize=8)
for i, v in enumerate(ivf.values[::-1]):
    ax.text(v + .01, i, f"{v:.3f}".replace(".", ","), va="center", fontsize=7)
for lim, rot in [(.02, "0,02"), (.1, "0,1"), (.3, "0,3"), (.5, "0,5")]:
    ax.axvline(lim, color=COR_CINZA, linestyle=":", linewidth=1)
    ax.text(lim, len(ivf) - .3, rot, fontsize=7, ha="center", va="bottom", color=COR_CINZA)
ax.set_xlim(0, 1.1); ax.set_ylim(-.6, len(ivf) + .2)
ax.set_xlabel("Information Value (IV)")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: f"{v:.1f}".replace(".", ",")))
fig.savefig(os.path.join(PASTA, "Figuras", "etapa5b_information_value.png"), facecolor="white"); plt.close(fig)
reg("  [figura] etapa5b_information_value.png")

# ---------------------------------------------------------------------------
NUM = ["recencia", "Frequencia_total", "TM_total"]
CAT = {"canais_cliente": "Multicanal", "status": "RETIDO", "recompra90": "Recomprou"}

def preparar(b):
    df, _ = carregar(b)
    tr, te = train_test_split(df, test_size=TESTE, stratify=df.churn, random_state=SEED, shuffle=True)
    reg("\n" + "=" * 78); reg(f"[4] DIVISAO TREINO/TESTE - {b}"); reg("=" * 78)
    reg(f"  total {len(df):,} | treino {len(tr):,} ({len(tr)/len(df)*100:.1f}%) | teste {len(te):,} ({len(te)/len(df)*100:.1f}%)".replace(",", "."))
    def resumo(d):
        return {"Taxa de churn (%)": d.churn.mean()*100,
                "Recência mediana (dias)": d.recencia.median(),
                "Frequência mediana": d.Frequencia_total.median(),
                "Frequência média": d.Frequencia_total.mean(),
                "Ticket mediano (R$)": d.TM_total.median(),
                "Valor acumulado mediano (R$)": d.VA_bruto_total.median(),
                "Multicanal (%)": (d.canais_cliente == "Multicanal").mean()*100,
                "Exclusivo varejo (%)": (d.canais_cliente == "Exclusivo Varejo").mean()*100,
                "Status retido (%)": (d.status == "RETIDO").mean()*100,
                "Status novo (%)": (d.status == "NOVO").mean()*100,
                "Recompra 90d - recomprou (%)": (d.recompra90 == "Recomprou").mean()*100,
                "Recompra 90d - janela incompleta (%)": (d.recompra90 == "Janela incompleta").mean()*100,
                "Idade valida (%)": d.idade_valida.mean()*100,
                "Outlier superior (%)": (d.flag_outlier == "Outlier Superior").mean()*100}
    ver = pd.DataFrame({"Treino": resumo(tr), "Teste": resumo(te), "Base": resumo(df)})
    reg("  Verificacao do equilibrio:"); reg("  " + ver.round(2).to_string().replace("\n", "\n  "))

    # idade: mediana do treino
    med = tr.loc[tr.idade_valida == 1, "idade_fim_2024"].median()
    reg(f"  mediana da idade (treino, registros validos) = {med:.0f} anos")
    for d in (tr, te):
        d["idade_ajustada"] = np.where(d.idade_valida == 1, d.idade_fim_2024, med)

    def matriz_arvore(d):
        X = d[["ClienteID", "churn"] + NUM + ["idade_ajustada", "idade_valida"]].copy()
        for c in CAT:
            for v in sorted(df[c].unique()):
                X[f"{c}={v}"] = (d[c] == v).astype(int)
        return X
    # logistica: ln + z com parametros do treino; dummies com categoria de referencia
    trans = {"recencia": tr.recencia, "ln_frequencia": np.log(tr.Frequencia_total),
             "ln_ticket": np.log(tr.TM_total), "idade_ajustada": tr.idade_ajustada}
    par = {k: (v.mean(), v.std(ddof=0)) for k, v in trans.items()}
    reg("  Parametros do escore-z (treino):")
    for k, (m, s) in par.items():
        reg(f"    {k:<16} media={m:.4f} desvio={s:.4f}")
    def matriz_log(d):
        X = d[["ClienteID", "churn"]].copy()
        base = {"recencia": d.recencia, "ln_frequencia": np.log(d.Frequencia_total),
                "ln_ticket": np.log(d.TM_total), "idade_ajustada": d.idade_ajustada}
        for k, v in base.items():
            X[f"z_{k}"] = (v - par[k][0]) / par[k][1]
        X["idade_valida"] = d.idade_valida.values
        for c, ref in CAT.items():
            for v in sorted(df[c].unique()):
                if v != ref:
                    X[f"{c}={v}"] = (d[c] == v).astype(int)
        return X
    reg("  Categorias de referencia (logistica): " + ", ".join(f"{c} = {r}" for c, r in CAT.items()))
    Lt, Le = matriz_log(tr), matriz_log(te)
    reg(f"  Matriz da logistica: {Lt.shape[1]-2} preditores -> {list(Lt.columns[2:])}")
    At = matriz_arvore(tr)
    reg(f"  Matriz da arvore: {At.shape[1]-2} preditores -> {list(At.columns[2:])}")
    # correlacao entre preditores da logistica (treino)
    C = Lt.iloc[:, 2:].corr()
    pares = [(a, b, C.loc[a, b]) for i, a in enumerate(C.columns) for b in C.columns[i+1:]]
    pares = sorted(pares, key=lambda x: -abs(x[2]))[:6]
    reg("  Maiores correlacoes entre preditores da logistica (treino, Pearson):")
    for a, b2, r in pares:
        reg(f"    {a} x {b2}: {r:+.3f}")
    for nome, X in [("treino_logistica", Lt), ("teste_logistica", Le),
                    ("treino_arvore", At), ("teste_arvore", matriz_arvore(te))]:
        X.to_csv(os.path.join(PASTA, f"5b_{b}_{nome}.csv"), index=False)
    reg(f"  [matrizes] 5b_{b}_*.csv (4 arquivos)")
    return ver

ver_t = preparar("base_total")
ver_s = preparar("base_sem_outliers")
open(os.path.join(PASTA, "numeros_etapa5b.txt"), "w", encoding="utf-8").write("\n".join(_saida) + "\n")
