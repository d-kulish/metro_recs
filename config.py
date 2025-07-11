"""Configuration settings for the recommendation pipeline."""

# BigQuery settings
PROJECT_ID = "cf-mccuagcf-recommenders-ht"
LOCATION = "EU"
DATASET_ID = "segmentation"

# --- Query Parameters ---
TRAINING_MONTH_ID = 202506
HISTORICAL_MONTH_ID = 202505
TARGET_CITY = "TERNOPIL"
TOP_PRODUCTS_PERCENTAGE = 0.81

# --- Reusable Query Fragments ---

# This CTE gets all relevant interactions for the training month and city.
# It also handles the casting of unsupported data types.
ALL_INTERACTIONS_CTE = f"""
AllInteractions AS (
  SELECT
    invoices.cust_person_id,
    -- Format DATE as STRING because BigQueryExampleGen doesn't support the DATE type.
    FORMAT_DATE('%Y-%m-%d', invoices.date_of_day) AS date_of_day,
    (invoices.art_no * 1000000 + invoices.var_tu_key) AS product_id,
    -- Cast NUMERIC to FLOAT64 as BigQueryExampleGen doesn't support NUMERIC.
    CAST(invoices.sell_val_nsp AS FLOAT64) AS sell_val_nsp,
    stores.city
  FROM
    `{PROJECT_ID}.{DATASET_ID}.ml_bi_invoices_tbl` AS invoices
  LEFT JOIN
    `{PROJECT_ID}.{DATASET_ID}.ml_bi_stores_tbl` AS stores
  ON
    invoices.store_id = stores.store_id
  WHERE
    invoices.month_id = {TRAINING_MONTH_ID}
    AND invoices.flag_cust_target_group = "SCO"
    AND sell_val_nsp > 0
    AND city = "{TARGET_CITY}"
)
"""

# This CTE calculates revenue per product from the interactions.
PRODUCT_REVENUES_CTE = """
ProductRevenues AS (
  SELECT
    product_id,
    SUM(sell_val_nsp) AS revenue
  FROM AllInteractions
  GROUP BY product_id
)
"""

# This CTE selects the top N% of products by revenue.
TOP_PRODUCTS_CTE = f"""
TopProducts AS (
  SELECT
    product_id
  FROM (
    SELECT
      product_id,
      revenue,
      SUM(revenue) OVER (ORDER BY revenue DESC) AS cumulative_revenue,
      (SELECT SUM(revenue) FROM ProductRevenues) AS total_revenue
    FROM ProductRevenues
  )
  WHERE cumulative_revenue - revenue < {TOP_PRODUCTS_PERCENTAGE} * total_revenue
)
"""

# --- Final Queries ---

# This query is used by the Trainer to get the list of all candidate products.
BQ_PRODUCTS_QUERY = f"""
WITH
{ALL_INTERACTIONS_CTE},
{PRODUCT_REVENUES_CTE},
{TOP_PRODUCTS_CTE}
SELECT product_id FROM TopProducts
"""

# This query is used by ExampleGen to get the training data.
BQ_QUERY = f"""
WITH
UserHistoricalRevenue AS (
  SELECT
    cust_person_id,
    -- Cast the summed revenue to FLOAT64 to avoid NUMERIC type errors in TFX.
    CAST(SUM(sell_val_nsp) AS FLOAT64) AS total_revenue
  FROM `{PROJECT_ID}.{DATASET_ID}.ml_bi_invoices_tbl`
  WHERE month_id = {HISTORICAL_MONTH_ID}
  GROUP BY cust_person_id
),
{ALL_INTERACTIONS_CTE},
{PRODUCT_REVENUES_CTE},
{TOP_PRODUCTS_CTE}
SELECT
  interactions.*,
  -- Use COALESCE to handle users who have no history, assigning them 0 revenue.
  COALESCE(hist.total_revenue, 0.0) AS total_revenue
FROM
  AllInteractions AS interactions
LEFT JOIN UserHistoricalRevenue AS hist ON interactions.cust_person_id = hist.cust_person_id
JOIN TopProducts ON interactions.product_id = TopProducts.product_id
"""

# Vertex AI settings
VERTEX_PROJECT_ID = PROJECT_ID
VERTEX_REGION = "europe-west4"
PIPELINE_NAME = "metro-recommendations-pipeline-gpu"
PIPELINE_ROOT = "gs://recs_metroua/pipeline_root"

# Dataflow settings for custom VPC. This is required if your project's network
# is in custom subnet mode. Ask your GCP admin for the correct subnetwork name
# for the 'europe-west4' region.
DATAFLOW_SUBNETWORK = "regions/europe-west4/subnetworks/default"

# Use a custom Docker image that includes tensorflow-recommenders.
# You must build and push this image to your project's Artifact Registry.
AR_REPO = "metro-recs-repo"  # The name of your Artifact Registry repository
IMAGE_NAME = "metro-recs-pipeline"
PIPELINE_IMAGE = f"{VERTEX_REGION}-docker.pkg.dev/{VERTEX_PROJECT_ID}/{AR_REPO}/{IMAGE_NAME}:latest-gpu"

# Service account settings - use the Vertex AI service account or the current compute account
# Option 1: Use the default Vertex AI service account (if it exists)
# VERTEX_SERVICE_ACCOUNT = f"vertex-pipelines@{PROJECT_ID}.iam.gserviceaccount.com"

# Option 2: Use the current compute service account (needs proper roles)
VERTEX_SERVICE_ACCOUNT = "1032729337493-compute@developer.gserviceaccount.com"

# Model settings - Updated for GPU training and scaling
EMBEDDING_DIMENSION = 32
TRAIN_EPOCHS = 5
TRAIN_STEPS = 1000  # Increased for better GPU utilization
EVAL_STEPS = 50  # Increased for better evaluation
BATCH_SIZE = 8192  # Larger batch size for GPU efficiency
LEARNING_RATE = 0.1

# GPU Training Configuration - Updated for modern Vertex AI
GPU_MACHINE_TYPE = "n1-standard-4"
GPU_ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
GPU_ACCELERATOR_COUNT = 1

# Distributed Training Configuration
ENABLE_DISTRIBUTED_TRAINING = False
WORKER_COUNT = 0  # 0 means no workers (single machine training)
PARAMETER_SERVER_COUNT = 0  # 0 means no parameter servers

# Scaling Configuration for 20-30x data increase
# Updated with modern GPU types and proper machine types
SCALING_CONFIG = {
    "large_dataset": {
        "GPU_MACHINE_TYPE": "n1-standard-8",
        "GPU_ACCELERATOR_TYPE": "NVIDIA_TESLA_T4",
        "GPU_ACCELERATOR_COUNT": 1,
        "BATCH_SIZE": 16384,
        "TRAIN_STEPS": 2000,
        "EVAL_STEPS": 100,
        "ENABLE_DISTRIBUTED_TRAINING": False,
        "WORKER_COUNT": 0,
        "PARAMETER_SERVER_COUNT": 0,
    },
    "extra_large_dataset": {
        "GPU_MACHINE_TYPE": "n1-standard-16",
        "GPU_ACCELERATOR_TYPE": "NVIDIA_TESLA_V100",
        "GPU_ACCELERATOR_COUNT": 1,
        "BATCH_SIZE": 32768,
        "TRAIN_STEPS": 3000,
        "EVAL_STEPS": 150,
        "ENABLE_DISTRIBUTED_TRAINING": False,
        "WORKER_COUNT": 0,
        "PARAMETER_SERVER_COUNT": 0,
    },
}


# Function to apply scaling configuration
def apply_scaling_config(config_name):
    """Apply scaling configuration for larger datasets."""
    if config_name in SCALING_CONFIG:
        globals().update(SCALING_CONFIG[config_name])
        print(f"Applied scaling configuration: {config_name}")
    else:
        print(f"Configuration {config_name} not found")
