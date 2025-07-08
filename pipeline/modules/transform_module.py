"""Transform module for Metro recommendations."""

import tensorflow as tf
import tensorflow_transform as tft

# All features that need to be transformed.
FEATURE_KEYS = ["cust_person_id", "product_id", "city", "date_of_day", "total_revenue"]

def transformed_name(key):
    """Generate transformed feature name."""
    return key + "_xf"


def preprocessing_fn(inputs):
    """tf.transform's callback function for preprocessing inputs.

    Args:
        inputs: map from feature keys to raw not-yet-transformed features.

    Returns:
        Map from string feature key to transformed feature operations.
    """
    outputs = {}

    # Create full vocabularies for all categorical features by including every
    # value that appears at least once.
    outputs[transformed_name("cust_person_id")] = tft.compute_and_apply_vocabulary(
        tf.strings.strip(tf.strings.as_string(inputs["cust_person_id"])),
        frequency_threshold=1,
        num_oov_buckets=1,
        vocab_filename="vocabulary_cust_person_id",
    )
    outputs[transformed_name("product_id")] = tft.compute_and_apply_vocabulary(
        tf.strings.strip(tf.strings.as_string(inputs["product_id"])),
        frequency_threshold=1,
        num_oov_buckets=1,
        vocab_filename="vocabulary_product_id",
    )
    outputs[transformed_name("city")] = tft.compute_and_apply_vocabulary(
        tf.strings.strip(tf.strings.as_string(inputs["city"])),
        frequency_threshold=1,
        num_oov_buckets=1,
        vocab_filename="vocabulary_city",
    )

    # Handle date feature by extracting cyclical components. This is robust
    # for future predictions as it captures seasonal patterns (e.g., month,
    # day) that are independent of the year.
    date_str = tf.strings.strip(tf.strings.as_string(inputs["date_of_day"]))

    # Extract month as a categorical feature (e.g., '06' from '2025-06-15').
    month_str = tf.strings.substr(date_str, 5, 2)
    outputs[transformed_name("month")] = tft.compute_and_apply_vocabulary(
        month_str, vocab_filename="vocabulary_month"
    )

    # Extract day of month as a categorical feature (e.g., '15' from '2025-06-15').
    day_of_month_str = tf.strings.substr(date_str, 8, 2)
    outputs[transformed_name("day_of_month")] = tft.compute_and_apply_vocabulary(
        day_of_month_str, vocab_filename="vocabulary_day_of_month"
    )

    # Handle the historical total_revenue feature.
    # 1. Normalize the revenue to a 0-1 range for use as a continuous feature.
    # 2. Bucketize the normalized revenue to create a categorical feature for embedding.
    normalized_revenue = tft.scale_to_0_1(
        tf.cast(inputs["total_revenue"], tf.float32)
    )
    outputs[transformed_name("total_revenue_normalized")] = normalized_revenue
    outputs[transformed_name("total_revenue_bucket")] = tft.bucketize(
        normalized_revenue, num_buckets=100
    )

    return outputs
