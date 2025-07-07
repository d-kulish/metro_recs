"""Configuration settings for the recommendation pipeline."""

# BigQuery settings
PROJECT_ID = "cf-mccuagcf-recommenders-ht"
LOCATION = "EU"
DATASET_ID = "segmentation"
BQ_QUERY = """
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
