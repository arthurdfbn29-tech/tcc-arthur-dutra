/* ============================================================================================================================
   QUERY COMPLETA - BASE DE CLIENTES (VERSAO ANONIMIZADA PARA REPOSITORIO PUBLICO)
   Segmento 1 / Vendas 2024 - 1 linha por cliente
   ---------------------------------------------------------------
   AVISO: nomes de projetos, datasets, tabelas e colunas de ORIGEM, ids internos de segmento e rotulos de canais
   foram substituidos por nomes genericos; a lista fixa de CPFs invalidos foi trocada por uma tabela de exclusoes.
   Os nomes das colunas de SAIDA (variaveis analiticas) foram mantidos. A logica de negocio (filtros, agregacoes,
   regras de status de cliente e de outlier) segue a da consulta original, de modo que a query documenta e permite
   reproduzir o metodo, mas nao executa sem adaptacao ao esquema de dados de quem a utilizar.
   Variantes analiticas testadas e descartadas (quebras por app/site, por colecao ON/OFF, CMV, valores liquidos de
   devolucao, quantidade liquida e NPS) foram removidas desta versao.

   CAMADA DE OUTLIERS:
     - flag_outlier             -> Outlier Superior / Outlier Inferior / Normal
     - percentil_rank_va_bruto  -> posicao relativa na distribuicao (0.00 a 1.00)
     - limite_inferior_p1       -> valor de corte P1  (rastreabilidade)
     - limite_superior_p99      -> valor de corte P99 (rastreabilidade)

   COMO USAR AS 3 BASES:
     Ao final do arquivo ha 3 blocos SELECT. Deixe ativo apenas o que deseja executar.
============================================================================================================================ */

WITH pesq_crm_1 AS ( -- pegar o primeiro pedido de 2024
  SELECT DISTINCT

  ClienteID,
  CASE
  WHEN LENGTH(LTRIM(RTRIM(crm.clienteid))) NOT IN (11, 14) THEN 'Invalido'
  WHEN crm.clienteid LIKE '%/%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%.%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%-%' THEN 'Invalido'
  WHEN REGEXP_CONTAINS(crm.clienteid, r'[a-zA-Z]') THEN 'Invalido'
  WHEN crm.clienteid LIKE '%012345678%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%12345678%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%9876543%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%000000%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%111111%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%222222%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%333333%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%444444%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%555555%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%666666%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%777777%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%888888%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%999999%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '00000%' THEN 'Invalido'
  WHEN crm.clienteid IN (SELECT cpf FROM `projeto.dataset_cadastro.tb_cpfs_invalidos`) THEN 'Invalido'
  ELSE
    'Valido'
  END AS check_cpf,
  chave_ordem,
  perfil_cliente,
  dt_ordem AS dt_transacao,

  FROM `projeto.dataset_crm.tb_crm_vendas` crm
  WHERE id_segmento = 1
  AND dt_transacao BETWEEN '2024-01-01' AND '2024-12-31'
  AND seq_ordem_ano = 1
),

pesq_crm_max AS ( -- pegar o ultimo pedido de 2024

  SELECT DISTINCT

  ClienteID,
  CASE
  WHEN LENGTH(LTRIM(RTRIM(crm.clienteid))) NOT IN (11, 14) THEN 'Invalido'
  WHEN crm.clienteid LIKE '%/%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%.%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%-%' THEN 'Invalido'
  WHEN REGEXP_CONTAINS(crm.clienteid, r'[a-zA-Z]') THEN 'Invalido'
  WHEN crm.clienteid LIKE '%012345678%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%12345678%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%9876543%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%000000%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%111111%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%222222%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%333333%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%444444%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%555555%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%666666%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%777777%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%888888%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '%999999%' THEN 'Invalido'
  WHEN crm.clienteid LIKE '00000%' THEN 'Invalido'
  WHEN crm.clienteid IN (SELECT cpf FROM `projeto.dataset_cadastro.tb_cpfs_invalidos`) THEN 'Invalido'
  ELSE
    'Valido'
  END AS check_cpf,
  chave_ordem,
  perfil_cliente,
  dt_ordem AS dt_transacao,

  FROM `projeto.dataset_crm.tb_crm_vendas` crm
  WHERE id_segmento = 1
  AND dt_transacao BETWEEN '2024-01-01' AND '2024-12-31'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY clienteid ORDER BY seq_ordem_ano DESC) = 1

),
pesq_crm_1_ant AS ( -- pegar o primeiro pedido de 2023
  SELECT DISTINCT

  ClienteID,
  chave_ordem,
  perfil_cliente,
  dt_ordem AS dt_transacao,

  FROM `projeto.dataset_crm.tb_crm_vendas` crm
  WHERE id_segmento = 1
  AND dt_transacao BETWEEN '2023-01-01' AND '2023-12-31'
  AND seq_ordem_ano = 1
),

pesq_crm_max_ant AS ( -- pegar o ultimo pedido de 2023

  SELECT DISTINCT

  ClienteID,
  chave_ordem,
  perfil_cliente,
  dt_ordem AS dt_transacao,

  FROM `projeto.dataset_crm.tb_crm_vendas` crm
  WHERE id_segmento = 1
  AND dt_transacao BETWEEN '2023-01-01' AND '2023-12-31'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY clienteid ORDER BY seq_ordem_ano DESC) = 1

),
pesq_crm_churn AS ( -- identificar se o cliente fez compra em 2025
  SELECT DISTINCT

  ClienteID,
  chave_ordem,
  perfil_cliente,

  FROM `projeto.dataset_crm.tb_crm_vendas` crm
  WHERE id_segmento = 1
  AND dt_transacao BETWEEN '2025-01-01' AND '2025-12-31'
  AND seq_ordem_ano = 1
),

pesq_crm_1compra AS ( -- pegar o primeira compra historica
  SELECT DISTINCT

  ClienteID,
  chave_ordem,
  canal_venda,

  FROM `projeto.dataset_crm.tb_crm_vendas` crm
  WHERE id_segmento = 1
  AND seq_ordem_hist = 1
),

pesq_crm_2024_total AS ( -- pegar a performance da pessoa no ano de 2024
  SELECT
  ClienteID,
  classificacao_canal,
  CASE WHEN VA_ecom >= VA_varejo THEN 'ECOMMERCE' ELSE 'VAREJO' END AS canal_favorito

  FROM(
  SELECT DISTINCT

  ClienteID,
  STRING_AGG(DISTINCT canal_venda ORDER BY canal_venda) AS canais,
  CASE
    WHEN STRING_AGG(DISTINCT canal_venda ORDER BY canal_venda) = 'ECOMMERCE'        THEN 'Exclusivo Ecommerce'
    WHEN STRING_AGG(DISTINCT canal_venda ORDER BY canal_venda) = 'VAREJO'           THEN 'Exclusivo Varejo'
    WHEN STRING_AGG(DISTINCT canal_venda ORDER BY canal_venda) = 'ECOMMERCE,VAREJO' THEN 'Multicanal'
    ELSE 'Sem Canal'
  END AS classificacao_canal,
  SUM(CASE WHEN canal_venda = 'ECOMMERCE' THEN vl_pago ELSE 0 END) AS VA_ecom,
  SUM(CASE WHEN canal_venda = 'VAREJO' THEN vl_pago ELSE 0 END) AS VA_varejo

  FROM `projeto.dataset_crm.tb_crm_vendas` crm
  WHERE id_segmento = 1
  AND dt_transacao BETWEEN '2024-01-01' AND '2024-12-31'
  GROUP BY ALL
  )
),

dados_cliente AS (
  SELECT
  cpf,
  sexo,
  DATE_DIFF(DATE('2024-12-31'), dt_nascimento, YEAR) -
    IF(EXTRACT(MONTH FROM dt_nascimento) > EXTRACT(MONTH FROM DATE('2024-12-31')) OR
      (EXTRACT(MONTH FROM dt_nascimento) = EXTRACT(MONTH FROM DATE('2024-12-31')) AND
       EXTRACT(DAY FROM dt_nascimento) > EXTRACT(DAY FROM DATE('2024-12-31'))), 1, 0) AS idade_fim_2024,
  uf AS estado,

  FROM `projeto.dataset_cadastro.tb_clientes`

  WHERE id_segmento_cad = 1
),
/* =========================================================================================
   BASE EXPLORATÓRIA POR CPF - Rede 2 / Vendas 2024
   -----------------------------------------------------------------------------------------
   - 1 linha por CPF
   - Filtros e regras de negócio mantidos EXATAMENTE como na consulta original.
   - Regras de canal:
        ecom   = channel = 'ECOMMERCE'
        app    = tipo_venda = 'App'
        site   = tipo_venda = 'Web'
        fisico = channel = 'FISICO'
     Para devoluções (returned), usa-se tipo_venda_devolucao_origi para identificar
     a origem da venda original.
   - Métricas: VA (bruto/líq), CMV (bruto/líq), Frequência, TM, Recompra 90/150/365,
               Devolução, Quantidade (bruta/líq), NPS (respondido/promotor/detrator/neutro).
   - Churn ignorado.
========================================================================================= */

troque AS (
  SELECT DISTINCT
       cod_ordem,
       ref_item,
       MIN(DATE(dt_solicitacao))  AS dt_solicitacao,
       MIN(dt_ocorrencia)             AS ultima_data_evento,
       status_final,
       `bigfunctions.us.remove_accents`(tp_reversa) AS tp_reversa_origem,
       CASE WHEN `bigfunctions.us.remove_accents`(tp_reversa) IN ('troca e devolucao','sem reembolso') THEN 'Troca'
            WHEN `bigfunctions.us.remove_accents`(tp_reversa) = 'devolucao' THEN 'Estorno'
            ELSE INITCAP(`bigfunctions.us.remove_accents`(tp_reversa)) END AS tp_reversa_tratado,
       CASE WHEN `bigfunctions.us.remove_accents`(motivo_rev) IN ('ficou diferente do que eu imaginava','desisti da compra','comprei o produto errado','escolhi um produto errado','desistencia','me arrependi da compra','nao gostei do produto','o produto nao me agradou','produto com cor diferente da imagem','produto nao corresponde com as fotos do site','desistir da compra') THEN 'Desistência'
            WHEN `bigfunctions.us.remove_accents`(motivo_rev) IN ('o tamanho ficou grande','ficou grande','tamanho ficou grande') THEN 'Ficou grande'
            WHEN `bigfunctions.us.remove_accents`(motivo_rev) IN ('o tamanho ficou pequeno','achei a peca muito curta','tamanho ficou pequeno','ficou pequeno') THEN 'Ficou pequeno'
            WHEN `bigfunctions.us.remove_accents`(motivo_rev) IN ('o produto nao veste bem','nao gostei do caimento da peca','medidas divergentes da tabela do site','nao gostei da modelagem','o tamanho nao ficou bom') THEN 'Modelagem'
            WHEN `bigfunctions.us.remove_accents`(motivo_rev) IN ('insucesso na entrega','recebi fora do prazo, nao tenho mais interesse','transportadora - atraso','transportadora - divergencia na entrega','transportadora - extravio','transportadora - pedido sendo devolvido','extravio') THEN 'Problema na entrega'
            WHEN `bigfunctions.us.remove_accents`(motivo_rev) IN ('a qualidade nao me agradou','nao gostei da qualidade do produto','veio com defeito','qualidade','defeitos e avarias','nao gostei da qualidade','produto com defeito','produto danificado','veio com defeito :(') THEN 'Qualidade/Defeito'
            WHEN `bigfunctions.us.remove_accents`(motivo_rev) IN ('recebi o produto errado','recebi meu produto incompleto','comprei uma peca e recebi outra','divergencia do produto','faltando acessorio','recebi divergente','recebi um produto errado','veio faltando acessorio','peca trocada','pendencia','produto errado','receber um produto errado') THEN 'Recebeu errado'
            ELSE 'Outros' END AS motivo_rev_tratado,
       CASE WHEN LOWER(transp) LIKE '%entrega pessoalmente%' THEN 'Vale-Troca em Loja'
            WHEN flag_troca_a = 1 THEN 'Troca PP'
            WHEN `bigfunctions.us.remove_accents`(tp_reversa) IN ('troca e devolucao','sem reembolso') THEN 'Troca'
            WHEN `bigfunctions.us.remove_accents`(tp_reversa) = 'devolucao' THEN 'Estorno'
            ELSE INITCAP(`bigfunctions.us.remove_accents`(tp_reversa)) END AS sub_tp_reversa_tratado
  FROM `projeto.dataset_pos_venda.tb_reversas`
  WHERE `bigfunctions.us.remove_accents`(status) = `bigfunctions.us.remove_accents`(LOWER(status_final))
  GROUP BY ALL
),

venda_origi AS (
  SELECT DISTINCT
       rvm.cod_ordem,
       CASE WHEN LOWER(rvm.politica_venda) LIKE '%app%' THEN 'APP'
            WHEN LOWER(rvm.politica_venda) LIKE '%canal_b%' THEN 'Canal Digital B'
            WHEN LOWER(rvm.politica_venda) LIKE '%programa fidelidade%' THEN 'Programa Fidelidade'
            WHEN LOWER(rvm.politica_venda) LIKE '%canal_c%' THEN 'Canal Digital C'
            WHEN LOWER(rvm.politica_venda) LIKE '%canal_d%' THEN 'Canal Digital D'
            WHEN LOWER(rvm.tp_operacao) LIKE '%online%' THEN 'Web'
            ELSE INITCAP(rvm.tp_operacao) END AS tp_operacao_dev_origem,
       CASE WHEN rvm.tp_operacao IN ('VITRINE','FRANQUIA','DEVOLUCAO') THEN INITCAP(rvm.tp_operacao)
            WHEN rs.nm_seller IS NOT NULL AND rs.filial_seller NOT LIKE '%ECOMM%' THEN 'Omnichannel'
            ELSE 'Online' END AS tipo_entrega_origi
  FROM `projeto.dataset_vendas.tb_vendas` rvm
  LEFT JOIN (
       SELECT * FROM (
         SELECT *, ROW_NUMBER() OVER (PARTITION BY cod_filial_seller ORDER BY nm_seller, filial_seller, dt_ini_seller, dt_fim_seller) AS rn
         FROM `projeto.dataset_vendas.tb_sellers`
       ) WHERE rn = 1
  ) rs
    ON  rs.nm_seller = rvm.ds_seller_fulfil
    AND rs.id_segmento_seller = rvm.id_segmento_canal
    AND DATE(rvm.dt_faturamento) >= rs.dt_ini_seller
    AND DATE(rvm.dt_faturamento) <  CASE WHEN rs.dt_fim_seller IS NULL THEN DATE_ADD(CURRENT_DATE(), INTERVAL 1000 YEAR) ELSE rs.dt_fim_seller END
  WHERE tp_operacao IN ('ONLINE','MARKETPLACE_PROPRIO','VITRINE')
    AND (
      (id_segmento_canal IN (1) AND DATE(dt_faturamento) BETWEEN '2021-08-01' AND CURRENT_DATE-1)
    )
    AND dt_ocorrencia >= '2020-12-01'
),

devolucoes AS (
  SELECT DISTINCT
       cod_ordem_orig,
       p.cod_prod,
       p.cod_var1,
       p.cod_var2,
       a.cod_ordem_orig || '_' || p.cod_prod || '_' || p.cod_var1 || '_' || p.cod_var2 AS chave
  FROM `projeto.dataset_vendas.tb_vendas` a
  LEFT JOIN `projeto.dataset_cadastro.tb_produtos` p
    ON p.id_sku = a.id_sku
  WHERE tp_operacao = 'DEVOLUCAO'
    AND (
      (id_segmento_canal IN (1) AND DATE(dt_faturamento) BETWEEN '2021-08-01' AND CURRENT_DATE-1)
    )
    AND dt_ocorrencia >= '2020-12-01'
),

------------------------------------------------------------------------------------------
-- BASE: nivel de item de venda (agregado por cliente nas etapas seguintes)
------------------------------------------------------------------------------------------
base_dados AS (
  SELECT
       dt_fat_ref,
       dt_ordem_ref,
       id_segmento_ref,
       id_segmento_produto,
       CASE WHEN status_ocorrencia = 'CANCELADO' THEN 'Vendas Canceladas'
            WHEN status_final   = 'FATURADO'  THEN 'Vendas Faturadas'
            END AS status,
       classe_origem,
       cupom_desc,
       canal_macro,
       CASE WHEN dt_desligamento < dt_ordem_ref THEN 'Vendedor Inativo' ELSE 'Vendedor Ativo' END AS status_vendedor,
       CASE WHEN vendedor IS NULL THEN 'Sem Vendedor' ELSE 'Com Código de Vendedor' END AS possui_cod_vendedor,
       INITCAP(item) AS item,
       grupo_colecao,
       marca_linha AS Estilista,
       CASE WHEN status_linha_prod = 'ON' THEN 'COLECAO'
            WHEN status_linha_prod = 'COLEÇÃO NÃO CADASTRADA' THEN 'OFF+SALE'
            ELSE 'OFF+SALE' END AS status_linha_prod,
       INITCAP(tp_operacao) AS tp_operacao,
       id_ordem,
       CASE WHEN INITCAP(tp_operacao) <> 'Devolucao' THEN INITCAP(tp_operacao)
            WHEN INITCAP(tp_operacao_dev_origem) IS NULL THEN 'Não Encontrado'
            ELSE INITCAP(tp_operacao_dev_origem) END AS tp_operacao_dev_origem,
       CASE WHEN INITCAP(tp_operacao) <> 'Devolucao' THEN INITCAP(tipo_entrega)
            WHEN INITCAP(tipo_entrega_origi) IS NULL THEN 'Não Encontrado'
            ELSE INITCAP(tipo_entrega_origi) END AS tipo_entrega_origi,
       tipo_entrega,
       INITCAP(ln_prod) AS ln_prod,
       INITCAP(grp_prod) AS grp_prod,
       ref_item_var AS ref_item_var,
       tam AS tam,
       doc_cliente,
       dt_solicitacao_rev,
       CASE WHEN status_final_reversa IN ('Finalizado','Cancelado') THEN NULL
            WHEN classe_origem = 'Venda Marketplace Interno' THEN NULL
            ELSE dt_ult_ocorrencia_rev END AS dt_ult_ocorrencia_rev,
       CASE WHEN status_final_reversa IN ('Finalizado','Cancelado') THEN NULL
            WHEN classe_origem = 'Venda Marketplace Interno' THEN NULL
            ELSE status_final_reversa END AS status_final_reversa,
       CASE WHEN dt_ordem_ref < '2023-01-01' THEN NULL ELSE tp_reversa_origem END AS tp_reversa_origem,
       CASE WHEN dt_ordem_ref < '2023-01-01' THEN NULL ELSE tp_reversa_tratado END AS tp_reversa_tratado,
       CASE WHEN dt_ordem_ref < '2023-01-01' THEN NULL ELSE sub_tp_reversa_tratado END AS sub_tp_reversa_tratado,
       CASE WHEN dt_ordem_ref < '2023-01-01' THEN NULL ELSE motivo_rev_tratado END AS motivo_rev_tratado,

       SUM(vl_desconto) AS vl_desconto,
       SUM(CASE WHEN qtde_pedidos = 1 AND INITCAP(tp_operacao) = 'Devolucao' THEN -qtde_pedidos
                WHEN qtde_pedidos = 1 AND status_final='FATURADO' AND tp_operacao_orig <>'DEVOLUCAO' AND status_ocorrencia='CANCELADO' THEN -qtde_pedidos
                WHEN qtde_pedidos = 1 THEN qtde_pedidos
                ELSE 0 END) AS qtde_pedidos,

       --- receita
       SUM(CASE WHEN status_final='FATURADO' AND tp_operacao_orig <>'DEVOLUCAO' AND status_ocorrencia='CAPTURADO' AND classe_origem <> 'Venda Marketplace Interno'
                THEN vl_pago_item END) AS total_payment_fulfilled,
       SUM(CASE WHEN status_final='FATURADO' AND tp_operacao_orig <>'DEVOLUCAO' AND status_ocorrencia='CANCELADO'
                THEN vl_pago_item END) AS total_payment_fulfilled_correcao,
       SUM(CASE WHEN tp_operacao_orig='DEVOLUCAO' THEN vl_pago_item END) AS total_payment_returned,

       --- cmv
       SUM(CASE WHEN status_final='FATURADO' AND tp_operacao_orig <>'DEVOLUCAO' AND status_ocorrencia='CAPTURADO' AND classe_origem <> 'Venda Marketplace Interno'
                THEN cmv END) AS cmv_fulfilled,
       SUM(CASE WHEN status_final='FATURADO' AND tp_operacao_orig <>'DEVOLUCAO' AND status_ocorrencia='CANCELADO'
                THEN cmv END) AS cmv_fulfilled_correcao,
       SUM(CASE WHEN tp_operacao_orig='DEVOLUCAO' THEN cmv END) AS cmv_returned,

       --- quantidade
       SUM(CASE WHEN status_final='FATURADO' AND tp_operacao_orig <>'DEVOLUCAO' AND status_ocorrencia='CAPTURADO' AND classe_origem <> 'Venda Marketplace Interno'
                THEN qtd_item END) AS total_quantity_fulfilled,
       SUM(CASE WHEN status_final='FATURADO' AND tp_operacao_orig <>'DEVOLUCAO' AND status_ocorrencia='CANCELADO'
                THEN qtd_item END) AS total_quantity_fulfilled_correcao,
       SUM(CASE WHEN tp_operacao_orig='DEVOLUCAO' THEN qtd_item END) AS total_quantity_returned

  FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY id_ordem, dt_fat_ref, id_segmento_ref, status_final ORDER BY dt_fat_ref) AS qtde_pedidos
    FROM (
      SELECT
            CASE WHEN rvm.tp_operacao <> 'DEVOLUCAO' AND rvm.cod_ordem LIKE '%fri%' THEN 98
                 WHEN rvm.tp_operacao  = 'DEVOLUCAO' AND rvm.cod_ordem_orig LIKE '%fri%' THEN 98
                 WHEN p.id_segmento_cad = 99 THEN 99
                 ELSE rvm.id_segmento_canal END AS id_segmento_ref,
            DATE(rvm.dt_faturamento) AS dt_fat_ref,
            CASE WHEN rvm.tp_operacao = 'DEVOLUCAO' AND dt_venda_orig IS NOT NULL THEN DATE(dt_venda_orig)
                 ELSE DATE(rvm.dt_venda) END AS dt_ordem_ref,
            CASE WHEN rvm.tp_operacao IN ('ONLINE','DEVOLUCAO','MARKETPLACE_PROPRIO') AND rvm.cod_ordem LIKE '%v%' THEN REGEXP_EXTRACT(rvm.cod_ordem, r'v[0-9]*[a-z]*')
                 WHEN rvm.tp_operacao IN ('VITRINE','DEVOLUCAO') AND rvm.cod_ordem LIKE '1-%' THEN rvm.cod_ordem
                 ELSE CONCAT(FORMAT_DATE('%Y-%m-%d', CASE WHEN dt_venda_orig IS NOT NULL THEN DATE(dt_venda_orig) ELSE DATE(rvm.dt_venda) END),
                             '_', cod_filial_canal, '_', rvm.cod_ordem) END AS id_ordem,
            rvm.cod_ordem,
            CASE WHEN rvm.cupom_desc IS NULL THEN 'Sem Cupom' ELSE rvm.cupom_desc END AS cupom_desc,
            CASE WHEN rvm.ds_seller_fulfil IS NULL THEN '-' ELSE rvm.ds_seller_fulfil END AS ds_seller_fulfil,
            CASE WHEN rs.filial_seller IS NOT NULL THEN rs.filial_seller
                 WHEN rs.desc_seller IS NOT NULL THEN rs.desc_seller
                 ELSE '-' END AS nome_filial_seller_fulfillment,
            CASE WHEN rvm.ds_seller_mktp IS NULL THEN '-' ELSE rvm.ds_seller_mktp END AS ds_seller_mktp,
            CASE WHEN rs_mktp.filial_seller IS NOT NULL THEN rs_mktp.filial_seller
                 WHEN rs_mktp.desc_seller IS NOT NULL THEN rs_mktp.desc_seller
                 ELSE '-' END AS nome_filial_seller_marketplace,
            rs_mktp.filial_seller,
            rs_mktp.desc_seller,
            rvm.status_final,
            rvm.margem,
            rvm.cmv,
            rvm.status_ocorrencia,
            rvm.vl_desconto,
            rvm.vl_pago_item,
            rvm.politica_venda,
            rvm.item AS id_produto,
            rvm.status_linha_prod,
            rvm.qtd_item,
            p.id_segmento_cad AS id_segmento_produto,

            CASE WHEN p.item IS NULL THEN 'Outros' ELSE p.item END AS item,
            grupo_colecao,
            p.cod_prod || '_' || p.cod_var1 AS ref_item_var,
            p.cod_var1 AS cod_var1,
            p.tam AS tam,
            p.cod_prod,
            p.vl_tabela,
            p.marca_linha,

            CASE WHEN rvm.tp_operacao = 'DEVOLUCAO' THEN t.dt_solicitacao ELSE NULL END AS dt_solicitacao_rev,
            CASE WHEN rvm.tp_operacao <> 'DEVOLUCAO' AND dev.cod_ordem_orig IS NULL AND DATE_DIFF(CURRENT_DATE-1, t.dt_solicitacao, DAY) <= 90
                 THEN t.ultima_data_evento ELSE NULL END AS dt_ult_ocorrencia_rev,
            CASE WHEN rvm.tp_operacao <> 'DEVOLUCAO' AND dev.cod_ordem_orig IS NULL AND DATE_DIFF(CURRENT_DATE-1, t.dt_solicitacao, DAY) <= 90
                 THEN t.status_final ELSE NULL END AS status_final_reversa,
            CASE WHEN rvm.tp_operacao = 'DEVOLUCAO' AND t.tp_reversa_origem IS NOT NULL THEN t.tp_reversa_origem
                 WHEN rvm.tp_operacao = 'DEVOLUCAO' AND rvm.ferramenta_origem = 'FERRAMENTA_A' THEN 'troca' ELSE NULL END AS tp_reversa_origem,
            CASE WHEN rvm.tp_operacao = 'DEVOLUCAO' AND t.tp_reversa_tratado IS NOT NULL THEN t.tp_reversa_tratado
                 WHEN rvm.tp_operacao = 'DEVOLUCAO' AND rvm.ferramenta_origem = 'FERRAMENTA_A' THEN 'Troca' ELSE NULL END AS tp_reversa_tratado,
            CASE WHEN rvm.tp_operacao = 'DEVOLUCAO' AND t.sub_tp_reversa_tratado IS NOT NULL THEN t.sub_tp_reversa_tratado
                 WHEN rvm.tp_operacao = 'DEVOLUCAO' AND rvm.ferramenta_origem = 'FERRAMENTA_A' THEN 'Vale-Troca em Loja' ELSE NULL END AS sub_tp_reversa_tratado,
            CASE WHEN rvm.tp_operacao = 'DEVOLUCAO' THEN t.motivo_rev_tratado ELSE NULL END AS motivo_rev_tratado,

            vo.tp_operacao_dev_origem,
            vo.tipo_entrega_origi,
            rvm.tp_operacao AS tp_operacao_orig,

            CASE WHEN LOWER(politica_venda) LIKE '%app%' THEN 'APP'
                 WHEN LOWER(politica_venda) LIKE '%canal_b%' THEN 'Canal Digital B'
                 WHEN LOWER(politica_venda) LIKE '%programa fidelidade%' THEN 'Programa Fidelidade'
                 WHEN LOWER(politica_venda) LIKE '%canal_c%' THEN 'Canal Digital C'
                 WHEN LOWER(politica_venda) LIKE '%canal_d%' THEN 'Canal Digital D'
                 WHEN LOWER(rvm.tp_operacao) LIKE '%online%' THEN 'Web'
                 ELSE INITCAP(rvm.tp_operacao) END AS tp_operacao,

            CASE WHEN rvm.tp_operacao IN ('ONLINE','MARKETPLACE_PROPRIO','VITRINE') THEN 'ECOMMERCE'
                 WHEN LOWER(politica_venda) LIKE '%app%' THEN 'ECOMMERCE'
                 WHEN rvm.tp_operacao = 'DEVOLUCAO' THEN 'DEVOLUCAO'
                 ELSE 'FISICO' END AS canal_macro,

            CASE WHEN rvm.tp_operacao IN ('VITRINE','FRANQUIA','DEVOLUCAO','FISICO','ESTOQUE PROPRIO','ESTOQUE PRÓPRIO') THEN INITCAP(rvm.tp_operacao)
                 WHEN rs.nm_seller IS NOT NULL AND rs.filial_seller NOT LIKE '%ECOMM%' THEN 'Omnichannel'
                 ELSE 'Online' END AS tipo_entrega,

            rvm.doc_cliente,
            CASE WHEN p.ln_prod IS NULL THEN 'outros' ELSE p.ln_prod END AS ln_prod,
            CASE WHEN p.grp_prod IS NULL THEN 'outros' ELSE p.grp_prod END AS grp_prod,
            v.dt_desligamento,
            v.dt_admissao_vend,
            rvm.cod_vendedor_evt AS vendedor,
            fo.tp_item AS Tipo_produto_fornecedor,
            fo.forn   AS forn,

            CASE WHEN LOWER(rvm.ds_seller_fulfil) LIKE '%mktp%' THEN 'Venda Marketplace Interno'
                 WHEN LOWER(rvm.ds_seller_fulfil) LIKE '%seller_interno_a%' THEN 'Venda Marketplace Interno'
                 WHEN LOWER(rvm.ds_seller_fulfil) LIKE '%seller_interno_b%' THEN 'Venda Marketplace Interno'
                 WHEN LOWER(rvm.ds_seller_fulfil) LIKE '%seller_interno_c%' THEN 'Venda Marketplace Interno'
                 WHEN id_segmento_canal = 2 AND LOWER(ds_seller_fulfil) IN ('seller_a','seller_b','seller_c') THEN 'Venda Marketplace Interno'
                 WHEN LOWER(rvm.ds_seller_mktp) IN ('mercadolivre','dafiti','amazon','marketplace_x')
                   OR LOWER(rvm.cod_ordem) LIKE '%amazon%'
                   OR LOWER(rvm.cod_ordem) LIKE '%mercadolivre%'
                   OR LOWER(rvm.cod_ordem) LIKE '%teste_integrador%'
                   OR LOWER(rvm.cod_ordem) LIKE '%integrador_x%' THEN 'Venda Marketplace Externo'
                 ELSE 'Venda Padrão' END AS classe_origem

      FROM `projeto.dataset_vendas.tb_vendas` rvm
      LEFT JOIN `projeto.dataset_cadastro.tb_vendedores` v
        ON v.cod_vend = rvm.cod_vendedor_evt
      LEFT JOIN (
           SELECT * FROM (
             SELECT *, ROW_NUMBER() OVER (PARTITION BY cod_filial_seller ORDER BY nm_seller, filial_seller, dt_ini_seller, dt_fim_seller) AS rn
             FROM `projeto.dataset_vendas.tb_sellers`
           ) WHERE rn = 1
      ) rs_mktp
        ON LOWER(rs_mktp.nm_seller) = LOWER(rvm.ds_seller_fulfil)
        AND rs_mktp.id_segmento_seller = rvm.id_segmento_canal
        AND DATE(rvm.dt_venda) >= CASE WHEN rs_mktp.dt_ini_seller IS NULL THEN DATE_SUB(CURRENT_DATE(), INTERVAL 1000 YEAR) ELSE rs_mktp.dt_ini_seller END
        AND DATE(rvm.dt_venda) <  CASE WHEN rs_mktp.dt_fim_seller    IS NULL THEN DATE_ADD(CURRENT_DATE(), INTERVAL 1000 YEAR) ELSE rs_mktp.dt_fim_seller    END
      LEFT JOIN (
           SELECT * FROM (
             SELECT *, ROW_NUMBER() OVER (PARTITION BY cod_filial_seller ORDER BY nm_seller, filial_seller, dt_ini_seller, dt_fim_seller) AS rn
             FROM `projeto.dataset_vendas.tb_sellers`
           ) WHERE rn = 1
      ) rs
        ON  rs.nm_seller = rvm.ds_seller_fulfil
        AND rs.id_segmento_seller = rvm.id_segmento_canal
        AND DATE(rvm.dt_faturamento) >= rs.dt_ini_seller
        AND DATE(rvm.dt_faturamento) <  CASE WHEN rs.dt_fim_seller IS NULL THEN DATE_ADD(CURRENT_DATE(), INTERVAL 1000 YEAR) ELSE rs.dt_fim_seller END
      LEFT JOIN `projeto.dataset_cadastro.tb_produtos` p
        ON p.id_sku = rvm.id_sku
      LEFT JOIN troque AS t
        ON t.cod_ordem ||'_'|| t.ref_item = (
             CASE WHEN rvm.tp_operacao = 'DEVOLUCAO'
                  THEN rvm.cod_ordem_orig || '_' || p.cod_prod || '_' || p.cod_var1 || '_' || p.cod_var2
                  ELSE rvm.cod_ordem         || '_' || p.cod_prod || '_' || p.cod_var1 || '_' || p.cod_var2 END
           )
      LEFT JOIN venda_origi AS vo
        ON vo.cod_ordem = rvm.cod_ordem_orig
      LEFT JOIN devolucoes AS dev
        ON dev.chave = rvm.cod_ordem || '_' || p.cod_prod || '_' || p.cod_var1 || '_' || p.cod_var2
      LEFT JOIN `projeto.dataset_cadastro.tb_fornecedores` AS fo
        ON p.cod_prod || '_' || p.id_segmento_cad = fo.cod_prod || '_' || fo.id_segmento_ref

      WHERE rvm.tp_operacao IN ('ONLINE','DEVOLUCAO','MARKETPLACE_PROPRIO','VITRINE','FISICO','ESTOQUE PROPRIO','ESTOQUE PRÓPRIO')
        AND (
             id_segmento_canal IN (1)
             AND (
                  CASE WHEN rvm.tp_operacao <> 'DEVOLUCAO'
                       THEN DATE(dt_faturamento)    BETWEEN '2024-01-01' AND '2024-12-31'
                       ELSE DATE(dt_venda_orig) BETWEEN '2024-01-01' AND '2024-12-31' END
             )
        )
        AND dt_ocorrencia >= '2020-12-01'
    )
  )
  WHERE classe_origem <> 'Venda Marketplace Externo'
  GROUP BY ALL
),

NPS AS (
  SELECT DISTINCT
       CASE WHEN tp_operacao IN ('Ecommerce','APP','Marketplace Proprio','Marketplace Proprio B','Programa Fidelidade','Canal Digital C','Canal Digital B') AND cod_ticket LIKE '%v%' THEN REGEXP_EXTRACT(cod_ticket, r'v[0-9]*[a-z]*')
            WHEN tp_operacao IN ('Vitrine') AND cod_ticket LIKE '1-%' THEN cod_ticket
            ELSE CONCAT(FORMAT_DATE('%Y-%m-%d', DATE(dt_ordem_ref)),
                             '_', CAST(cod_filial AS INT), '_', cod_ticket) END AS id_ordem,
       nota_pesq,
       status_pesq
  FROM `projeto.dataset_pos_venda.tb_pesquisa_satisfacao`
  WHERE id_segmento = 1

),

------------------------------------------------------------------------------------------
-- Primeira compra de cada CPF (para Recompra 90/150/365 dias)
------------------------------------------------------------------------------------------
primeira_compra AS (
  SELECT
       doc_cliente,
       MIN(dt_ordem_ref) AS dt_primeira_compra
  FROM base_dados
  WHERE status = 'Vendas Faturadas'
    AND tp_operacao <> 'Devolucao'
    AND classe_origem <> 'Venda Marketplace Interno'
  GROUP BY doc_cliente
),

------------------------------------------------------------------------------------------
-- NPS POR CPF
------------------------------------------------------------------------------------------
nps_por_cpf AS (
  SELECT
       b.doc_cliente,

       COUNT(DISTINCT b.id_ordem) AS NPS_respondido_total,
       COUNT(DISTINCT CASE WHEN b.canal_macro    = 'ECOMMERCE' THEN b.id_ordem END) AS NPS_respondido_ecom,
       COUNT(DISTINCT CASE WHEN b.tp_operacao = 'App'       THEN b.id_ordem END) AS NPS_respondido_app,
       COUNT(DISTINCT CASE WHEN b.tp_operacao = 'Web'       THEN b.id_ordem END) AS NPS_respondido_site,
       COUNT(DISTINCT CASE WHEN b.canal_macro    = 'FISICO'    THEN b.id_ordem END) AS NPS_respondido_fisico,

       COUNT(DISTINCT CASE WHEN n.status_pesq = 'Promotor'                                         THEN b.id_ordem END) AS NPS_promotor_total,
       COUNT(DISTINCT CASE WHEN n.status_pesq = 'Promotor' AND b.canal_macro    = 'ECOMMERCE'         THEN b.id_ordem END) AS NPS_promotor_ecom,
       COUNT(DISTINCT CASE WHEN n.status_pesq = 'Promotor' AND b.tp_operacao = 'App'               THEN b.id_ordem END) AS NPS_promotor_app,
       COUNT(DISTINCT CASE WHEN n.status_pesq = 'Promotor' AND b.tp_operacao = 'Web'               THEN b.id_ordem END) AS NPS_promotor_site,
       COUNT(DISTINCT CASE WHEN n.status_pesq = 'Promotor' AND b.canal_macro    = 'FISICO'            THEN b.id_ordem END) AS NPS_promotor_fisico,

       COUNT(DISTINCT CASE WHEN n.status_pesq = 'Detrator'                                         THEN b.id_ordem END) AS NPS_detrator_total,
       COUNT(DISTINCT CASE WHEN n.status_pesq = 'Detrator' AND b.canal_macro    = 'ECOMMERCE'         THEN b.id_ordem END) AS NPS_detrator_ecom,
       COUNT(DISTINCT CASE WHEN n.status_pesq = 'Detrator' AND b.tp_operacao = 'App'               THEN b.id_ordem END) AS NPS_detrator_app,
       COUNT(DISTINCT CASE WHEN n.status_pesq = 'Detrator' AND b.tp_operacao = 'Web'               THEN b.id_ordem END) AS NPS_detrator_site,
       COUNT(DISTINCT CASE WHEN n.status_pesq = 'Detrator' AND b.canal_macro    = 'FISICO'            THEN b.id_ordem END) AS NPS_detrator_fisico,

       COUNT(DISTINCT CASE WHEN n.status_pesq = 'Neutro'                                           THEN b.id_ordem END) AS NPS_neutro_total,
       COUNT(DISTINCT CASE WHEN n.status_pesq = 'Neutro'   AND b.canal_macro    = 'ECOMMERCE'         THEN b.id_ordem END) AS NPS_neutro_ecom,
       COUNT(DISTINCT CASE WHEN n.status_pesq = 'Neutro'   AND b.tp_operacao = 'App'               THEN b.id_ordem END) AS NPS_neutro_app,
       COUNT(DISTINCT CASE WHEN n.status_pesq = 'Neutro'   AND b.tp_operacao = 'Web'               THEN b.id_ordem END) AS NPS_neutro_site,
       COUNT(DISTINCT CASE WHEN n.status_pesq = 'Neutro'   AND b.canal_macro    = 'FISICO'            THEN b.id_ordem END) AS NPS_neutro_fisico

  FROM base_dados b
  INNER JOIN NPS n ON n.id_ordem = b.id_ordem
  WHERE b.status = 'Vendas Faturadas'
    AND b.tp_operacao <> 'Devolucao'
    AND b.classe_origem <> 'Venda Marketplace Interno'
    AND b.doc_cliente IS NOT NULL
  GROUP BY b.doc_cliente
),

------------------------------------------------------------------------------------------
-- SELECT FINAL: 1 linha por CPF
------------------------------------------------------------------------------------------

NPS_e_valor AS(
SELECT
    b.doc_cliente AS CPF,

    /* ============================================================================
       VA BRUTO (Total / Ecom / App / Site / Físico)
       ============================================================================ */
    SUM(IFNULL(b.total_payment_fulfilled,0)) + SUM(IFNULL(b.total_payment_fulfilled_correcao,0)) AS VA_bruto_total,
    SUM(CASE WHEN b.canal_macro = 'ECOMMERCE' THEN IFNULL(b.total_payment_fulfilled,0) ELSE 0 END)
      + SUM(CASE WHEN b.canal_macro = 'ECOMMERCE' THEN IFNULL(b.total_payment_fulfilled_correcao,0) ELSE 0 END) AS VA_bruto_ecom,

    SUM(CASE WHEN b.canal_macro = 'FISICO' THEN IFNULL(b.total_payment_fulfilled,0) ELSE 0 END)
      + SUM(CASE WHEN b.canal_macro = 'FISICO' THEN IFNULL(b.total_payment_fulfilled_correcao,0) ELSE 0 END) AS VA_bruto_fisico,


    COUNT(DISTINCT CASE WHEN b.status = 'Vendas Faturadas' AND b.tp_operacao <> 'Devolucao' THEN b.id_ordem END) AS Frequencia_total,
    COUNT(DISTINCT CASE WHEN b.status = 'Vendas Faturadas' AND b.tp_operacao <> 'Devolucao' AND b.canal_macro    = 'ECOMMERCE' THEN b.id_ordem END) AS Frequencia_ecom,

    COUNT(DISTINCT CASE WHEN b.status = 'Vendas Faturadas' AND b.tp_operacao <> 'Devolucao' AND b.canal_macro    = 'FISICO'    THEN b.id_ordem END) AS Frequencia_fisico,

    /* ============================================================================
       TICKET MÉDIO (TM) = VA Bruto / Frequência
       ============================================================================ */
    SAFE_DIVIDE(
      SUM(IFNULL(b.total_payment_fulfilled,0)) + SUM(IFNULL(b.total_payment_fulfilled_correcao,0)),
      COUNT(DISTINCT CASE WHEN b.status='Vendas Faturadas' AND b.tp_operacao <> 'Devolucao' THEN b.id_ordem END)
    ) AS TM_total,
    SAFE_DIVIDE(
      SUM(CASE WHEN b.canal_macro='ECOMMERCE' THEN IFNULL(b.total_payment_fulfilled,0) ELSE 0 END)
        + SUM(CASE WHEN b.canal_macro='ECOMMERCE' THEN IFNULL(b.total_payment_fulfilled_correcao,0) ELSE 0 END),
      COUNT(DISTINCT CASE WHEN b.status='Vendas Faturadas' AND b.tp_operacao <> 'Devolucao' AND b.canal_macro='ECOMMERCE' THEN b.id_ordem END)
    ) AS TM_ecom,

    SAFE_DIVIDE(
      SUM(CASE WHEN b.canal_macro='FISICO' THEN IFNULL(b.total_payment_fulfilled,0) ELSE 0 END)
        + SUM(CASE WHEN b.canal_macro='FISICO' THEN IFNULL(b.total_payment_fulfilled_correcao,0) ELSE 0 END),
      COUNT(DISTINCT CASE WHEN b.status='Vendas Faturadas' AND b.tp_operacao <> 'Devolucao' AND b.canal_macro='FISICO' THEN b.id_ordem END)
    ) AS TM_fisico,

    /* ============================================================================
       RECOMPRA 90 / 150 / 365 DIAS
       ============================================================================ */
    MAX(CASE WHEN b.dt_ordem_ref >  pc.dt_primeira_compra
                AND b.dt_ordem_ref <= DATE_ADD(pc.dt_primeira_compra, INTERVAL 90 DAY)
                AND b.status = 'Vendas Faturadas' AND b.tp_operacao <> 'Devolucao' AND b.classe_origem <> 'Venda Marketplace Interno'
             THEN 1 ELSE 0 END) AS Recompra_90_dias,
    MAX(CASE WHEN b.dt_ordem_ref >  pc.dt_primeira_compra
                AND b.dt_ordem_ref <= DATE_ADD(pc.dt_primeira_compra, INTERVAL 150 DAY)
                AND b.status = 'Vendas Faturadas' AND b.tp_operacao <> 'Devolucao' AND b.classe_origem <> 'Venda Marketplace Interno'
             THEN 1 ELSE 0 END) AS Recompra_150_dias,
    MAX(CASE WHEN b.dt_ordem_ref >  pc.dt_primeira_compra
                AND b.dt_ordem_ref <= DATE_ADD(pc.dt_primeira_compra, INTERVAL 365 DAY)
                AND b.status = 'Vendas Faturadas' AND b.tp_operacao <> 'Devolucao' AND b.classe_origem <> 'Venda Marketplace Interno'
             THEN 1 ELSE 0 END) AS Recompra_365_dias,

    /* ============================================================================
       DEVOLUÇÃO (Total / ON / OFF)
       ============================================================================ */

    SUM(IFNULL(b.total_quantity_fulfilled,0)) + SUM(IFNULL(b.total_quantity_fulfilled_correcao,0)) AS Qtd_bruta_total,
    SUM(CASE WHEN  b.canal_macro='ECOMMERCE'  THEN IFNULL(b.total_quantity_fulfilled,0) ELSE 0 END)
      + SUM(CASE WHEN  b.canal_macro='ECOMMERCE'  THEN IFNULL(b.total_quantity_fulfilled_correcao,0) ELSE 0 END) AS Qtd_bruta_ecom,
    SUM(CASE WHEN b.canal_macro='FISICO' THEN IFNULL(b.total_quantity_fulfilled,0) ELSE 0 END)
      + SUM(CASE WHEN b.canal_macro='FISICO' THEN IFNULL(b.total_quantity_fulfilled_correcao,0) ELSE 0 END) AS Qtd_bruta_OFF,

    /* ============================================================================
       QUANTIDADE LÍQUIDA (Total / ON / OFF)
       ============================================================================ */

FROM base_dados b
LEFT JOIN primeira_compra pc USING (doc_cliente)
LEFT JOIN nps_por_cpf      np USING (doc_cliente)
WHERE b.doc_cliente IS NOT NULL
GROUP BY b.doc_cliente
),

/* ============================================================================================================================
   NOVA CTE 1 - base_final
   Encapsula o SELECT original (1 linha por CPF) para permitir
   que as CTEs de outlier referenciem os dados sem reprocessamento.
============================================================================================================================ */
base_final AS (
  SELECT
    a.ClienteID,
    CASE
      WHEN a.perfil_cliente = 'NOVO'           AND b.perfil_cliente = 'RETIDO' THEN 'NOVO RETRONANTE'
      WHEN a.perfil_cliente LIKE '%REATIVADO%' AND b.perfil_cliente = 'RETIDO' THEN 'REATIVADO RETORNANTE'
      WHEN a.perfil_cliente LIKE '%REATIVADO%'                               THEN 'REATIVADO'
      ELSE a.perfil_cliente
    END AS status_cliente_ano,
    a.dt_transacao                                    AS data_1_pedido_2024,
    b.dt_transacao                                    AS data_ultimo_pedido_2024,
    c.dt_transacao                                    AS data_1_pedido_2023,
    d.dt_transacao                                    AS data_ultimo_pedido_2023,
    CASE WHEN e.ClienteID IS NULL THEN 1 ELSE 0 END AS churn,
    f.canal_venda                                        AS canal_entrada_historico,
    g.classificacao_canal                          AS canais_cliente,
    g.canal_favorito,
    h.* EXCEPT (cpf),
    i.* EXCEPT (cpf)

  FROM pesq_crm_1 a
  LEFT JOIN pesq_crm_max        b ON a.ClienteID = b.ClienteID
  LEFT JOIN pesq_crm_1_ant      c ON a.ClienteID = c.ClienteID
  LEFT JOIN pesq_crm_max_ant    d ON a.ClienteID = d.ClienteID
  LEFT JOIN pesq_crm_churn      e ON a.ClienteID = e.ClienteID
  LEFT JOIN pesq_crm_1compra    f ON a.ClienteID = f.ClienteID
  LEFT JOIN pesq_crm_2024_total g ON a.ClienteID = g.ClienteID
  LEFT JOIN dados_cliente       h ON a.ClienteID = h.cpf
  LEFT JOIN NPS_e_valor         i ON a.ClienteID = i.cpf

  -- DEPOIS
  WHERE a.check_cpf = 'Valido'
  AND i.VA_bruto_total IS NOT NULL   -- remove nulos
  AND i.VA_bruto_total > 0           -- remove negativos e zeros
),

/* ============================================================================================================================
   NOVA CTE 2 - limites_percentil
   Calcula P1 e P99 de VA_bruto_total.
   PERCENTILE_CONT é analítica no BigQuery; QUALIFY garante 1 linha de saída.
   Ajuste 0.01 / 0.99 para fronteiras mais conservadoras (ex.: 0.05 / 0.95).
============================================================================================================================ */
limites_percentil AS (
  SELECT
    PERCENTILE_CONT(VA_bruto_total, 0.01) OVER () AS p1,    -- Limite inferior (1º percentil)
    PERCENTILE_CONT(VA_bruto_total, 0.99) OVER () AS p99    -- Limite superior (99º percentil)
  FROM base_final
  WHERE VA_bruto_total IS NOT NULL
  QUALIFY ROW_NUMBER() OVER () = 1
),

/* ============================================================================================================================
   NOVA CTE 3 - base_classificada
   Une todos os campos originais com as colunas de outlier:
     - flag_outlier              -> Outlier Superior / Outlier Inferior / Normal
     - percentil_rank_va_bruto   -> posição relativa na distribuição (0.00 a 1.00)
     - limite_inferior_p1        -> valor de corte P1  (rastreabilidade metodológica)
     - limite_superior_p99       -> valor de corte P99 (rastreabilidade metodológica)
============================================================================================================================ */
base_classificada AS (
  SELECT
    bf.*,

    -- Limites de referência (mantidos para auditoria e documentação no TCC)
    lp.p1   AS limite_inferior_p1,
    lp.p99  AS limite_superior_p99,

    -- Posição percentual do CPF na distribuição de VA_bruto_total (0 = menor, 1 = maior)
    ROUND(
      PERCENT_RANK() OVER (ORDER BY bf.VA_bruto_total),
      4
    ) AS percentil_rank_va_bruto,

    -- Classificação principal de outlier
    CASE
      WHEN bf.VA_bruto_total > lp.p99 THEN 'Outlier Superior'   -- Gasto muito acima do usual
      WHEN bf.VA_bruto_total < lp.p1  THEN 'Outlier Inferior'   -- Gasto muito abaixo do usual
      ELSE                                  'Normal'             -- Massa principal de clientes
    END AS flag_outlier,

    NTILE(4) OVER (ORDER BY bf.VA_bruto_total)         AS quartil_va_bruto,

  FROM base_final bf
  CROSS JOIN limites_percentil lp   -- seguro: limites_percentil retorna exatamente 1 linha
)

/* ============================================================================================================================
   SELECTS FINAIS
   Deixe ATIVO apenas o bloco desejado. Comente os outros dois antes de executar.
   ============================================================================================================================

   BASE 1 - TOTAL: todos os CPFs + classificação de outlier
   ---------------------------------------------------------- */
--SELECT *
--FROM base_classificada
--ORDER BY VA_bruto_total DESC

----------------------------------------------------------
   --BASE 2 - FILTRADA POR flag_outlier (ativo: apenas 'Normal'; troque o WHERE para isolar os outliers)

SELECT DISTINCT *
FROM base_classificada
--WHERE flag_outlier IN ('Outlier Superior')
WHERE flag_outlier = 'Normal'
ORDER BY flag_outlier, VA_bruto_total DESC
;

/* ----------------------------------------------------------
   BASE 3 - TRATADA: apenas clientes 'Normal' (sem outliers)
   Base limpa recomendada para entrada nos algoritmos K-Means / RFM

SELECT *
FROM base_classificada
WHERE flag_outlier = 'Normal'
ORDER BY VA_bruto_total DESC
;
*/