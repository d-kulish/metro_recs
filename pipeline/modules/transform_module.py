"""Transform module for preprocessing BigQuery data."""

import tensorflow as tf
import tensorflow_transform as tft

NUM_OOV_BUCKETS = 1


def preprocessing_fn(inputs):
    """Preprocessing function for BigQuery data."""
    outputs = {}

    # Process user_id and product_id
    outputs["user_id"] = tft.sparse_tensor_to_dense_with_shape(
        inputs["user_id"], [None, 1], "-1"
    )
    outputs["product_id"] = tft.sparse_tensor_to_dense_with_shape(
        inputs["product_id"], [None, 1], "-1"
    )

    # Create vocabularies for user_id and product_id
    tft.compute_and_apply_vocabulary(
        inputs["user_id"],
        num_oov_buckets=NUM_OOV_BUCKETS,
        vocab_filename="user_id_vocab",
    )

    tft.compute_and_apply_vocabulary(
        inputs["product_id"],
        num_oov_buckets=NUM_OOV_BUCKETS,
        vocab_filename="product_id_vocab",
    )

    # Optional: include other features like city, sell_val_nsp
    if "city" in inputs:
        outputs["city"] = tft.sparse_tensor_to_dense_with_shape(
            inputs["city"], [None, 1], "-1"
        )
        tft.compute_and_apply_vocabulary(
            inputs["city"], num_oov_buckets=NUM_OOV_BUCKETS, vocab_filename="city_vocab"
        )

    return outputs
