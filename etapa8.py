"""
Etapa 8 - Integracao das Frentes A e B e base das acoes propostas.

Cruza os clusters do K-means (Etapa 6A.1/7A, base total) com:
  1. o churn observado e o valor acumulado de 2024 perdido (base inteira);
  2. o desempenho da regressao logistica e da arvore (1 EP) no conjunto de
     TESTE, por cluster, com o limiar q da Etapa 7B;
  3. recortes que sustentam cada acao: C2 por status e canal, C3 por recencia
     e frequencia, clientes de altissimo valor (Outlier Superior) e C1.
Nenhum modelo e reajustado.

Uso: python etapa8.py [pasta]   Saida: numeros_etapa8.txt
"""
import sys, os
import numpy as np, pandas as pd
from sklearn.metrics import precision_score, recall_score

PASTA = sys.argv[1] if len(sys.argv) > 1 else "."
_s = []
def reg(t=""):
    print(t); _s.append(str(t))
br = lambda v, d=1: f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
mi = lambda v: f"R$ {br(v/1e6)} mi"

d = pd.read_csv(os.path.join(PASTA, "base_total.csv"))
d = d.merge(pd.read_csv(os.path.join(PASTA, "rotulos_kmeans_base_total.csv")), on="ClienteID")
d["C"] = "C" + d.cluster_kmeans.astype(str)
d["recencia"] = (pd.Timestamp("2024-12-31") - pd.to_datetime(d.data_ultimo_pedido_2024)).dt.days
d["status"] = d.status_cliente_ano.replace({"NOVO RETRONANTE": "NOVO RETORNANTE"})
assert d.ClienteID.is_unique

reg("=" * 78); reg("ETAPA 8 - INTEGRACAO CLUSTERS x CHURN x MODELOS (base total)"); reg("=" * 78)
receita = d.VA_bruto_total.sum(); perd = d.loc[d.churn == 1, "VA_bruto_total"].sum()
reg(f"receita 2024: {mi(receita)} | valor das clientes que saíram: {mi(perd)} ({br(perd/receita*100)}%)")
reg(); reg("1) VALOR PERDIDO POR CLUSTER")
for c, g in d.groupby("C"):
    ch = g[g.churn == 1]
    reg(f"  {c}: n={br(len(g),0)} churn={br(g.churn.mean()*100)}% saíram={br(len(ch),0)} "
        f"valor perdido={mi(ch.VA_bruto_total.sum())} ({br(ch.VA_bruto_total.sum()/perd*100)}% do perdido) "
        f"mediana VA perdidas=R$ {br(ch.VA_bruto_total.median(),0)}")

reg(); reg("2) DESEMPENHO POR CLUSTER NO TESTE (limiar q = churn do treino)")
q = pd.read_csv(os.path.join(PASTA, "5b_base_total_treino_arvore.csv"), usecols=["churn"]).churn.mean()
t = pd.read_csv(os.path.join(PASTA, "6b1_probabilidades_base_total.csv"))
t = t.merge(pd.read_csv(os.path.join(PASTA, "6b2_probabilidades_base_total_1EP.csv"))[["ClienteID", "prob_arvore"]], on="ClienteID")
t = t.merge(d[["ClienteID", "C", "VA_bruto_total"]], on="ClienteID")
reg(f"  teste n={br(len(t),0)} | q={q:.4f}")
for col, nome in [("prob_logistica", "logística"), ("prob_arvore", "árvore 1EP")]:
    reg(f"  -- {nome}")
    for c, g in t.groupby("C"):
        y = g.churn.values; yh = (g[col].values > q).astype(int); w = g.VA_bruto_total.values
        rv = (w * (y == 1) * (yh == 1)).sum() / (w * (y == 1)).sum()
        reg(f"    {c}: n={br(len(g),0)} churn={br(y.mean()*100)}% recall={br(recall_score(y,yh)*100)}% "
            f"precisão={br(precision_score(y,yh,zero_division=0)*100)}% recall_valor={br(rv*100)}% "
            f"sinalizadas={br(yh.mean()*100)}%")

reg(); reg("3) C2 - STATUS E CANAL (ação 1)")
c2 = d[d.C == "C2"]
reg((c2.groupby("status").churn.agg(n="size", churn=lambda s: round(s.mean()*100, 1))).to_string())
reg(pd.crosstab(c2.status, c2.canais_cliente, values=c2.churn, aggfunc="mean").mul(100).round(1).to_string())
reg(pd.crosstab(c2.status, c2.canais_cliente).to_string())
nv = c2[c2.status == "NOVO"]
for k, g in nv.groupby("canais_cliente"):
    reg(f"  NOVO C2 {k}: n={br(len(g),0)} churn={br(g.churn.mean()*100)}% valor perdido={mi(g.loc[g.churn==1,'VA_bruto_total'].sum())} TM mediano=R$ {br(g.TM_total.median(),0)}")
reg(f"  NOVO C2 total: n={br(len(nv),0)} valor perdido={mi(nv.loc[nv.churn==1,'VA_bruto_total'].sum())} ({br(nv.loc[nv.churn==1,'VA_bruto_total'].sum()/perd*100)}% do perdido total)")
for k, g in c2.groupby("canais_cliente"):
    reg(f"  C2 {k}: n={br(len(g),0)} churn={br(g.churn.mean()*100)}%")

reg(); reg("4) C3 - RECÊNCIA E FREQUÊNCIA (ação 2)")
c3 = d[d.C == "C3"]
faixas = pd.cut(c3.recencia, [-1, 30, 60, 90, 120, 180, 400], labels=["0-30", "31-60", "61-90", "91-120", "121-180", ">180"])
reg(c3.groupby(faixas, observed=True).churn.agg(n="size", churn=lambda s: round(s.mean()*100, 1)).to_string())
reg(f"  >60 dias: n={br((c3.recencia>60).sum(),0)} ({br((c3.recencia>60).mean()*100)}% do C3) churn={br(c3.loc[c3.recencia>60,'churn'].mean()*100)}% | <=60: churn={br(c3.loc[c3.recencia<=60,'churn'].mean()*100)}%")
reg(f"  recência mediana: saíram {c3.loc[c3.churn==1,'recencia'].median():.0f} d | ficaram {c3.loc[c3.churn==0,'recencia'].median():.0f} d")
reg(f"  frequência mediana: saíram {c3.loc[c3.churn==1,'Frequencia_total'].median():.0f} | ficaram {c3.loc[c3.churn==0,'Frequencia_total'].median():.0f}")
ff = pd.cut(c3.Frequencia_total, [1, 3, 5, 10, 20, 1000], labels=["2-3", "4-5", "6-10", "11-20", ">20"])
reg(c3.groupby(ff, observed=True).agg(n=("churn", "size"), churn=("churn", lambda s: round(s.mean()*100, 1)),
    receita_pct=("VA_bruto_total", lambda s: round(s.sum()/receita*100, 1))).to_string())

reg(); reg("5) CLIENTES DE ALTÍSSIMO VALOR - Outlier Superior (ação 3)")
v = d[d.flag_outlier == "Outlier Superior"]
reg(f"  n={len(v)} | no C3: {(v.C=='C3').sum()} | receita {br(v.VA_bruto_total.sum()/receita*100)}% | churn {br(v.churn.mean()*100)}% ({v.churn.sum()} clientes, {mi(v.loc[v.churn==1,'VA_bruto_total'].sum())})")
reg(f"  F mediana {v.Frequencia_total.median():.0f} | TM mediano R$ {br(v.TM_total.median(),0)} | VA mediano R$ {br(v.VA_bruto_total.median(),0)}")
reg(f"  canais: {v.canais_cliente.value_counts().to_dict()} | favorito: {v.canal_favorito.value_counts().to_dict()}")
reg(f"  frequência física mediana {v.Frequencia_fisico.median():.0f} | % VA em loja física {br(v.VA_bruto_fisico.sum()/v.VA_bruto_total.sum()*100)}%")
reg(f"  recência mediana: saíram {v.loc[v.churn==1,'recencia'].median():.0f} d | ficaram {v.loc[v.churn==0,'recencia'].median():.0f} d | >30 d: {br((v.recencia>30).mean()*100)}% das clientes")

reg(); reg("6) C1 (ação 4)")
c1 = d[d.C == "C1"]
f1 = pd.cut(c1.recencia, [-1, 180, 240, 300, 400], labels=["<=180", "181-240", "241-300", ">300"])
reg(c1.groupby(f1, observed=True).churn.agg(n="size", churn=lambda s: round(s.mean()*100, 1)).to_string())
reg(f"  TM mediano R$ {br(c1.TM_total.median(),0)} | VA mediano R$ {br(c1.VA_bruto_total.median(),0)} | canais {c1.canais_cliente.value_counts(normalize=True).mul(100).round(1).to_dict()}")
reg(f"  status: {c1.status.value_counts(normalize=True).mul(100).round(1).to_dict()}")

with open(os.path.join(PASTA, "numeros_etapa8.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(_s) + "\n")
