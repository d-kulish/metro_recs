"""Transform module for Metro recommendations."""

import tensorflow as tf
import tensorflow_transform as tft

# Feature names
LABEL_KEY = "sell_val_nsp"
FEATURE_KEYS = ["cust_person_id", "product_id", "city"]


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

    # Transform categorical features
    for key in FEATURE_KEYS:
        if key in inputs:
            outputs[transformed_name(key)] = tft.compute_and_apply_vocabulary(
                tf.strings.strip(tf.strings.as_string(inputs[key])),
                top_k=10000,
                num_oov_buckets=1,
                # Explicitly name the vocabulary file to match the trainer's expectation.
                vocab_filename=f"vocabulary_{key}",
            )

    # Transform label if present
    if LABEL_KEY in inputs:
        outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.float32)

    return outputs
