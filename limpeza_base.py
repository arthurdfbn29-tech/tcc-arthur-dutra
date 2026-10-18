"""
Etapa 3 - Validacao e limpeza da base (TCC).
Le a aba 'Base ativa completa', aplica as regras de limpeza descritas na Etapa 3
do Capitulo 4 e imprime o funil e os numeros citados no texto.

Uso:  python limpeza_base.py "Dados tcc - Versão Claude.xlsx"

Regras aplicadas (todas sobre a base inteira, antes de separar as duas bases):
  1. deduplica clientes com ClienteID repetido, mantendo 1 linha por cliente
     (ver escolher_linha_dedup: prioriza status "Novo Retornante"/"Reativado
     Retornante"; no empate, prioriza a linha cuja data_1_pedido_2023 cai de
     fato em 2023);
  2. remove linhas com Qtd_bruta_fisico < 0 (cobre também Qtd_bruta_total < 0,
     que é subconjunto) -- tratado como sujeira de lançamento;
  3. remove linhas com Qtd_bruta_total == 0;
  4. corrige (sem remover linha) os 501 casos em que canais_cliente diverge
     da frequência por canal: zera Frequencia, VA_bruto e Qtd_bruta do canal
     "extra" (o que a classificação já não reconhecia), e ajusta
     Frequencia_total na mesma proporção. Justificativa: são lançamentos
     residuais de devolução/troca em loja que não geram compra líquida;
  5. remove o resíduo de valor negativo em canal que sobra após o passo 4
     (clientes Multicanal com VA negativo em um canal, não cobertos pela
     correção acima);
  6. remove as linhas com inconsistência de datas (último pedido antes do
     primeiro, ou primeiro pedido de 2024 registrado em dezembro/2023);
  7. remove apenas a coluna sexo (as colunas de outlier -- limite_inferior_p1,
     limite_superior_p99, percentil_rank_va_bruto, quartil_va_bruto -- foram
     mantidas a pedido do autor, para uso em etapas futuras);
  8. cria idade_valida e estado_valido (indicadores binarios, recodificacao
     direta -- SEM imputar). idade_ajustada (a imputacao) fica para a
     Etapa 6B; estado nao e imputado (categorica, sem estatistica central
     aplicavel), so ganha o indicador.
NAO altera: canais_cliente (a classificação em si).

Duas bases analiticas, usadas nas DUAS frentes (A - clusterizacao e
B - previsao de churn), para comparar o efeito dos outliers nos modelos:
  base_sem_outliers.csv   -> apenas flag_outlier == 'Normal'
  base_total.csv          -> todos (normais + inferiores + superiores)

O ClienteID (CPF) e substituido por um hash antes de qualquer saida (LGPD).
NAO versione os CSVs nem o .xlsx no GitHub.
"""
import sys, os, hashlib
import numpy as np
import pandas as pd

ARQ = sys.argv[1] if len(sys.argv) > 1 else "Dados tcc - Versão Claude.xlsx"
ABA = "Base ativa completa"
# Sal do hash: lido da variavel de ambiente TCC_SAL, nunca escrito no codigo.
# Ex.: (PowerShell) $env:TCC_SAL="seu-texto"  |  (bash) export TCC_SAL="seu-texto"
SAL = os.environ.get("TCC_SAL")
if not SAL:
    sys.exit("Defina a variavel de ambiente TCC_SAL (sal do hash do ClienteID) antes de executar.")

df = pd.read_excel(ARQ, sheet_name=ABA)
df = df.rename(columns={"Qtd_bruta_OFF": "Qtd_bruta_fisico"})   # tolera o nome antigo
rho = df[["VA_bruto_total", "percentil_rank_va_bruto", "quartil_va_bruto"]].corr(method="spearman")
df["ClienteID"] = df["ClienteID"].astype(str).map(
    lambda x: hashlib.sha256((SAL + x).encode()).hexdigest()[:16])
funil = [("Base extraída (Etapa 2)", 0, len(df))]


def escolher_linha_dedup(g):
    """Recebe as linhas (>=2) de um mesmo ClienteID duplicado e devolve o
    índice da linha a manter, seguindo a regra definida com o autor:
      a) se alguma linha tem status "Novo Retornante" ou "Reativado
         Retornante" (a fonte grava "RETRONANTE"), restringe aos candidatos
         com esse status;
      b) dentro dos candidatos, prioriza a linha cuja data_1_pedido_2023 cai
         de fato no ano de 2023 (exclui datas-sentinela como 2022-12-30);
      c) se ainda houver empate, mantém a linha com data_1_pedido_2024 mais
         antiga (critério de desempate, não deve ocorrer nos 31 casos
         observados)."""
    status = g["status_cliente_ano"].astype(str).str.upper()
    retornante = status.str.contains("RETRONANTE") | status.str.contains("RETORNANTE")
    cand = g[retornante] if retornante.any() else g
    d23 = pd.to_datetime(cand["data_1_pedido_2023"])
    em_2023 = (d23 >= "2023-01-01") & (d23 <= "2023-12-31")
    cand2 = cand[em_2023] if em_2023.any() else cand
    if len(cand2) > 1:
        cand2 = cand2.sort_values("data_1_pedido_2024").iloc[[0]]
    return cand2.index[0]


# 1. Unicidade -- deduplicar mantendo 1 linha por cliente (não descartar o cliente)
dup_mask = df.duplicated("ClienteID", keep=False)
n_clientes_dup = df.loc[dup_mask, "ClienteID"].nunique()
idx_manter = df[dup_mask].groupby("ClienteID").apply(escolher_linha_dedup, include_groups=False)
linhas_removidas = dup_mask.sum() - len(idx_manter)
print(f"[1] IDs duplicados: {n_clientes_dup} | linhas removidas: {linhas_removidas} "
      f"(mantida 1 linha por cliente, nenhum cliente é descartado)")
df = pd.concat([df[~dup_mask], df.loc[idx_manter.tolist()]]).sort_index().copy()
funil.append(("Deduplicação (mantendo 1 linha por cliente)", int(linhas_removidas), len(df)))

# 2. Quantidade bruta negativa (físico ou total -- sujeira de lançamento)
neg = (df.Qtd_bruta_fisico < 0) | (df.Qtd_bruta_total < 0)
print(f"[2] Qtd_bruta_fisico<0 ou Qtd_bruta_total<0: {neg.sum()} | por grupo: {df[neg].flag_outlier.value_counts().to_dict()}")
df = df[~neg].copy()
funil.append(("Remoção de quantidade bruta negativa (físico ou total)", int(neg.sum()), len(df)))

# 2b. Quantidade bruta total igual a zero
zero = df.Qtd_bruta_total == 0
print(f"[2b] Qtd_bruta_total == 0: {zero.sum()} | por grupo: {df[zero].flag_outlier.value_counts().to_dict()}")
df = df[~zero].copy()
funil.append(("Remoção de quantidade bruta total igual a zero", int(zero.sum()), len(df)))

# 3. Diagnosticos (NADA e alterado)
print(f"[3] Ausentes (%): sexo={df.sexo.isna().mean()*100:.1f} | idade={df.idade_fim_2024.isna().mean()*100:.1f}"
      f" | estado={df.estado.isna().sum()} linhas | TM_ecom={df.TM_ecom.isna().mean()*100:.1f}"
      f" | TM_fisico={df.TM_fisico.isna().mean()*100:.1f} | data_1_pedido_2023={df.data_1_pedido_2023.isna().mean()*100:.1f}")
inv = (df.idade_fim_2024 < 10) | (df.idade_fim_2024 > 100)
print(f"    idade inválida (<10 ou >100): {inv.sum()} | 114 anos: {(df.idade_fim_2024==114).sum()} | 124 anos: {(df.idade_fim_2024==124).sum()}")

# 3b. Tratamento da idade (parcial, nesta etapa): cria APENAS o indicador
#     idade_valida, que e uma recodificacao direta (nao depende de nenhuma
#     estatistica estimada sobre a base, entao nao ha risco de vazamento).
#     idade_ajustada (a imputacao pela mediana) NAO e criada aqui -- fica
#     para a Etapa 6B, depois da divisao treino/teste, para que a mediana
#     usada seja calculada so com o conjunto de treino.
df["idade_valida"] = (~df.idade_fim_2024.isna() & ~inv).astype(int)
print(f"    idade_valida=1: {df.idade_valida.sum()} ({df.idade_valida.mean()*100:.1f}%) | "
      f"idade_valida=0: {(df.idade_valida==0).sum()} ({(df.idade_valida==0).mean()*100:.1f}%) "
      f"(idade_ajustada fica para a Etapa 6B, pos train/test split)")

# 3c. Mesmo indicador para estado (categorica -- sem imputacao por mediana;
#     so o indicador de ausencia, que ja e o suficiente para o modelo saber
#     que o dado nao existe).
df["estado_valido"] = (~df.estado.isna()).astype(int)
print(f"    estado_valido=1: {df.estado_valido.sum()} ({df.estado_valido.mean()*100:.1f}%) | "
      f"estado_valido=0: {(df.estado_valido==0).sum()} ({(df.estado_valido==0).mean()*100:.1f}%)")

# 4. Corrige a divergencia canais_cliente x frequencia por canal (NAO remove
#    linha; zera o canal "extra" e ajusta Frequencia_total na mesma
#    proporcao, para manter a identidade Frequencia_total = ecom + fisico).
mask_so_ecom = (df.canais_cliente == "Exclusivo Ecommerce") & (df.Frequencia_fisico > 0)
mask_so_varejo = (df.canais_cliente == "Exclusivo Varejo") & (df.Frequencia_ecom > 0)
n_div = int(mask_so_ecom.sum() + mask_so_varejo.sum())
df.loc[mask_so_ecom, "Frequencia_total"] -= df.loc[mask_so_ecom, "Frequencia_fisico"]
df.loc[mask_so_ecom, ["Frequencia_fisico", "VA_bruto_fisico", "Qtd_bruta_fisico"]] = 0
df.loc[mask_so_varejo, "Frequencia_total"] -= df.loc[mask_so_varejo, "Frequencia_ecom"]
df.loc[mask_so_varejo, ["Frequencia_ecom", "VA_bruto_ecom", "Qtd_bruta_ecom"]] = 0
print(f"[4] canais_cliente x frequência corrigidos: {n_div} "
      f"(Exclusivo Ecommerce c/ freq. física: {int(mask_so_ecom.sum())} | "
      f"Exclusivo Varejo c/ freq. ecom: {int(mask_so_varejo.sum())}) -- canal extra zerado, Frequencia_total ajustada")

# 4b. Recomposicao dos totais APOS a correcao acima.
#     Zerar um canal torna os totais herdados da query inconsistentes com as
#     proprias componentes: a Frequencia_total ja foi ajustada no passo 4, mas
#     VA, quantidade e ticket medio continuariam com os valores antigos. Como
#     TM = VA / Frequencia por definicao, manter o TM herdado quebraria essa
#     identidade justamente nos clientes corrigidos. Recompoem-se os totais a
#     partir das componentes por canal, apenas nas linhas corrigidas.
_corrigidos = mask_so_ecom | mask_so_varejo
_tm_antes = df.TM_total.copy()
_va_antes = df.VA_bruto_total.copy()
df.loc[_corrigidos, "VA_bruto_total"] = (df.loc[_corrigidos, "VA_bruto_ecom"]
                                         + df.loc[_corrigidos, "VA_bruto_fisico"])
df.loc[_corrigidos, "Qtd_bruta_total"] = (df.loc[_corrigidos, "Qtd_bruta_ecom"]
                                          + df.loc[_corrigidos, "Qtd_bruta_fisico"])
df.loc[_corrigidos, "TM_total"] = (df.loc[_corrigidos, "VA_bruto_total"]
                                   / df.loc[_corrigidos, "Frequencia_total"].replace(0, np.nan))
df.loc[mask_so_ecom, "TM_fisico"] = np.nan     # canal zerado: ticket indefinido
df.loc[mask_so_varejo, "TM_ecom"] = np.nan
_dtm = int(((_tm_antes - df.TM_total).abs() > 0.01).sum())
_dva = float((_va_antes - df.VA_bruto_total).sum())
print(f"[4b] Totais recompostos nas {int(_corrigidos.sum())} linhas corrigidas: "
      f"TM alterado em {_dtm} | receita total reduzida em R$ {_dva:,.2f}")
print(f"     mediana do TM: R$ {_tm_antes.median():,.2f} -> R$ {df.TM_total.median():,.2f}")

# 5. Resíduo de valor negativo em canal que sobra após a correção acima
#    (ex.: cliente Multicanal com VA negativo em um canal -- não é coberto
#    pela regra do passo 4, que só trata os casos "Exclusivo").
resid = (df.VA_bruto_ecom < 0) | (df.VA_bruto_fisico < 0) | (df.Qtd_bruta_ecom < 0) | (df.Qtd_bruta_fisico < 0)
print(f"[5] Resíduo de valor/quantidade negativa em canal (pós passo 4): {resid.sum()}")
df = df[~resid].copy()
funil.append(("Remoção de resíduo negativo em canal (após correção da divergência)", int(resid.sum()), len(df)))

# 6. Inconsistência de datas: último pedido antes do primeiro, ou primeiro
#    pedido de 2024 registrado em dezembro/2023 -- removidos por instrução
#    do autor, para não introduzir ruído nas variáveis de recência/tempo
#    entre pedidos usadas pelos modelos.
datas_inconsistentes = (df.data_ultimo_pedido_2024 < df.data_1_pedido_2024) | (df.data_1_pedido_2024 < "2024-01-01")
print(f"[6] Inconsistência de datas (último<primeiro ou primeiro em dez/2023): {datas_inconsistentes.sum()}")
df = df[~datas_inconsistentes].copy()
funil.append(("Remoção de inconsistência de datas", int(datas_inconsistentes.sum()), len(df)))

# 7. Colunas removidas -- só sexo (as colunas de outlier foram mantidas)
cols_fora = ["sexo"]
p1, p99 = df.limite_inferior_p1.iloc[0], df.limite_superior_p99.iloc[0]
print(f"[7] P1 = R$ {p1:,.2f} | P99 = R$ {p99:,.2f} | Spearman VA x rank={rho.iloc[0,1]:.2f}, VA x quartil={rho.iloc[0,2]:.2f}"
      f" (colunas de outlier MANTIDAS na base)")
df = df.drop(columns=cols_fora)

# 5. Perfil dos grupos de outlier (base ja limpa)
print("\n[5] Perfil por grupo (base limpa)")
tot = df.VA_bruto_total.sum()
for k, g in df.groupby("flag_outlier"):
    print(f"    {k}: n={len(g)} | churn={g.churn.mean()*100:.1f}% (churn=1: {int(g.churn.sum())})"
          f" | mediana VA=R$ {g.VA_bruto_total.median():,.2f} | mediana freq={g.Frequencia_total.median():.0f}"
          f" | ticket mediano=R$ {g.TM_total.median():,.2f} | receita={g.VA_bruto_total.sum()/tot*100:.2f}%"
          f" | multicanal={(g.canais_cliente=='Multicanal').mean()*100:.0f}%")

# 6. Duas bases analiticas (usadas nas duas frentes, A e B)
normal = df.flag_outlier == "Normal"
bases = {"base_sem_outliers": df[normal], "base_total": df}
print("\nFUNIL")
for nome, saiu, resta in funil:
    print(f"  {nome:<50} -{saiu:>5}  -> {resta:>6}")
print("\nBASES ANALÍTICAS")
for nome, b in bases.items():
    print(f"  {nome:<22} {len(b):>6} clientes x {b.shape[1]} colunas | churn={b.churn.mean()*100:.1f}%"
          f" | receita={b.VA_bruto_total.sum()/tot*100:.1f}% da base total")
    b.to_csv(f"{nome}.csv", index=False)
