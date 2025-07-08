"""Configuration settings for the recommendation pipeline."""

# BigQuery settings
PROJECT_ID = "cf-mccuagcf-recommenders-ht"
LOCATION = "EU"
DATASET_ID = "segmentation"
BQ_QUERY = """
WITH UserHistoricalRevenue AS (
  SELECT
    cust_person_id,
    SUM(sell_val_nsp) AS total_revenue
  FROM `cf-mccuagcf-recommenders-ht.segmentation.ml_bi_invoices_tbl`
  WHERE month_id = 202505 -- The month *before* the training data
  GROUP BY cust_person_id
),
AllInteractions AS (
  SELECT
    invoices.cust_person_id,
    invoices.date_of_day,
    (invoices.art_no * 1000000 + invoices.var_tu_key) AS product_id,
    invoices.sell_val_nsp,
    stores.city
  FROM
    `cf-mccuagcf-recommenders-ht.segmentation.ml_bi_invoices_tbl` AS invoices
  LEFT JOIN
    `cf-mccuagcf-recommenders-ht.segmentation.ml_bi_stores_tbl` AS stores
  ON
    invoices.store_id = stores.store_id
  WHERE
    invoices.month_id = 202506
    AND invoices.flag_cust_target_group = "SCO"
    AND sell_val_nsp > 0
    AND city = "TERNOPIL"
),
ProductRevenues AS (
  SELECT
    product_id,
    SUM(sell_val_nsp) AS revenue
  FROM AllInteractions
  GROUP BY product_id
),
TopProducts AS (
  SELECT
    product_id
  FROM (
    SELECT
      product_id,
      revenue,
      SUM(revenue) OVER (ORDER BY revenue DESC) AS cumulative_revenue
    FROM ProductRevenues
  )
  WHERE cumulative_revenue - revenue < 0.8 * (SELECT SUM(revenue) FROM ProductRevenues)
)
SELECT
  interactions.*,
  -- Use COALESCE to handle users who have no history, assigning them 0 revenue.
  COALESCE(hist.total_revenue, 0.0) AS total_revenue
FROM
  AllInteractions AS interactions
LEFT JOIN UserHistoricalRevenue AS hist ON interactions.cust_person_id = hist.cust_person_id
JOIN TopProducts ON interactions.product_id = TopProducts.product_id
"""

BQ_PRODUCTS_QUERY = """
WITH ProductRevenues AS (
  SELECT
    (invoices.art_no * 1000000 + invoices.var_tu_key) AS product_id,
    SUM(invoices.sell_val_nsp) AS revenue
  FROM
    `cf-mccuagcf-recommenders-ht.segmentation.ml_bi_invoices_tbl` AS invoices
  LEFT JOIN
    `cf-mccuagcf-recommenders-ht.segmentation.ml_bi_stores_tbl` AS stores
  ON
    invoices.store_id = stores.store_id
  WHERE
    invoices.month_id = 202506
    AND invoices.flag_cust_target_group = "SCO"
    AND invoices.sell_val_nsp > 0
    AND stores.city = "TERNOPIL"
  GROUP BY
    product_id
)
SELECT
  product_id
FROM (
  SELECT
    product_id,
    revenue,
    SUM(revenue) OVER (ORDER BY revenue DESC) AS cumulative_revenue
  FROM ProductRevenues
)
WHERE cumulative_revenue - revenue < 0.8 * (SELECT SUM(revenue) FROM ProductRevenues)
"""
# Vertex AI settings
VERTEX_PROJECT_ID = PROJECT_ID
VERTEX_REGION = "europe-west4"
PIPELINE_NAME = "metro-recommendations-pipeline"
PIPELINE_ROOT = "gs://recs_metroua/pipeline_root"

# Model settings
EMBEDDING_DIMENSION = 32
TRAIN_EPOCHS = 5
TRAIN_STEPS = 500
EVAL_STEPS = 10
