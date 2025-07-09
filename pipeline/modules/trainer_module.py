"""Trainer module for Metro recommendations model."""

from typing import Dict, List, Text
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_recommenders as tfrs
from absl import logging

from tfx import v1 as tfx
from tfx_bsl.public import tfxio

# We will query for candidate data directly in the trainer.
from google.cloud import bigquery

# Define constants and helpers directly in this module to avoid
# problematic cross-module imports in the TFX execution environment.
LABEL_KEY = "sell_val_nsp"

def transformed_name(key):
    """Generate transformed feature name."""
    return key + "_xf"

EMBEDDING_DIMENSION = 32


class MetroRecommendationModel(tfrs.Model):
    """Metro recommendation model using TFX."""

    def __init__(self, user_model, product_model):
        super().__init__()
        self.product_model = product_model
        self.user_model = user_model

        # Initialize the task without metrics. They will be attached later in the
        # run_fn after the candidate dataset has been created.
        self.task = tfrs.tasks.Retrieval()

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        """Compute loss for training."""
        # Create a dictionary of only the features needed by the user model
        # to avoid passing unnecessary data like the product_id.
        user_features = {
            key: value
            for key, value in features.items()
            if key != transformed_name("product_id")
        }

        user_embeddings = self.user_model(user_features)
        positive_product_embeddings = self.product_model(
            features[transformed_name("product_id")]
        )
        return self.task(
            user_embeddings, positive_product_embeddings, compute_metrics=not training
        )


def _input_fn(
    file_pattern: List[str],
    data_accessor: tfx.components.DataAccessor,
    tf_transform_output: tft.TFTransformOutput,
    batch_size: int = 256,
) -> tf.data.Dataset:
    """Generates features and label for training."""
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size,
            # A retrieval task does not need a label.
            label_key=None,
        ),
        schema=tf_transform_output.transformed_metadata.schema,
    )


def _build_user_model(
    tf_transform_output: tft.TFTransformOutput,
) -> tf.keras.Model:
    """Builds the user model tower with multiple features."""
    inputs = {
        transformed_name("cust_person_id"): tf.keras.Input(shape=(1,), name=transformed_name("cust_person_id"), dtype=tf.int64),
        transformed_name("city"): tf.keras.Input(shape=(1,), name=transformed_name("city"), dtype=tf.int64),
        transformed_name("month"): tf.keras.Input(shape=(1,), name=transformed_name("month"), dtype=tf.int64),
        transformed_name("day_of_month"): tf.keras.Input(shape=(1,), name=transformed_name("day_of_month"), dtype=tf.int64),
        transformed_name("total_revenue_bucket"): tf.keras.Input(shape=(1,), name=transformed_name("total_revenue_bucket"), dtype=tf.int64),
        transformed_name("total_revenue_normalized"): tf.keras.Input(shape=(1,), name=transformed_name("total_revenue_normalized"), dtype=tf.float32),
    }

    embeddings = []

    # Create embeddings for all categorical features
    for key, vocab_name, num_buckets in [
        (transformed_name("cust_person_id"), "vocabulary_cust_person_id", None),
        (transformed_name("city"), "vocabulary_city", None),
        (transformed_name("month"), "vocabulary_month", None),
        (transformed_name("day_of_month"), "vocabulary_day_of_month", None),
        (transformed_name("total_revenue_bucket"), None, 100),
    ]:
        # The input_dim must be vocab_size + num_oov_buckets. In our transform, num_oov_buckets is 1.
        # This fixes the "index out of bounds" error.
        vocab_size = (num_buckets + 1) if num_buckets else (tf_transform_output.vocabulary_size_by_name(vocab_name) + 1)
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=EMBEDDING_DIMENSION, name=f"embedding_{key}"
        )
        embeddings.append(tf.keras.layers.Flatten()(embedding_layer(inputs[key])))

    # Add the normalized revenue directly as a feature
    embeddings.append(inputs[transformed_name("total_revenue_normalized")])

    # Concatenate all features and build dense layers for the final embedding
    concatenated_features = tf.keras.layers.concatenate(embeddings)
    dense_output = tf.keras.layers.Dense(64, activation="relu")(
        concatenated_features
    )
    dense_output = tf.keras.layers.Dense(EMBEDDING_DIMENSION)(dense_output)

    return tf.keras.Model(inputs=inputs, outputs=dense_output)

def _build_product_model(
    tf_transform_output: tft.TFTransformOutput,
) -> tf.keras.Model:
    """Builds the product model tower."""
    # The input_dim must be vocab_size + num_oov_buckets. In our transform, num_oov_buckets is 1.
    vocab_size = tf_transform_output.vocabulary_size_by_name("vocabulary_product_id") + 1
    product_id_input = tf.keras.Input(
        shape=(1,), name=transformed_name("product_id"), dtype=tf.int64
    )
    embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=EMBEDDING_DIMENSION, name="embedding_product_id"
    )(product_id_input)
    return tf.keras.Model(inputs=product_id_input, outputs=tf.keras.layers.Flatten()(embedding))


def run_fn(fn_args: tfx.components.FnArgs):
    """Main training function called by TFX Trainer."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    tft_layer = tf_transform_output.transform_features_layer()

    # Create datasets
    train_dataset = _input_fn(
        fn_args.train_files, fn_args.data_accessor, tf_transform_output, 4096
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, fn_args.data_accessor, tf_transform_output, 4096
    )

    # Build model
    model = MetroRecommendationModel(
        user_model=_build_user_model(tf_transform_output),
        product_model=_build_product_model(tf_transform_output),
    )

    # Compile and train
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq="batch"
    )

    # --- Candidate set creation for evaluation ---
    logging.info("Querying BigQuery for candidate products...")
    project_id = fn_args.custom_config["project_id"]
    products_query = fn_args.custom_config["products_query"]
    client = bigquery.Client(project=project_id)
    products_df = client.query(products_query).to_dataframe()
    logging.info(f"Found {len(products_df)} candidate products.")

    # Create a tf.data.Dataset of raw product IDs.
    products_ds = tf.data.Dataset.from_tensor_slices(
        products_df["product_id"].astype(str)
    )

    # Create a dataset of dictionaries for preprocessing.
    products_dict_ds = products_ds.map(lambda x: {"product_id": x})

    # Create a dedicated preprocessing layer for product IDs. This is crucial
    # because the main `tft_layer` expects all features, but here we only have
    # the product_id. We use the vocabulary computed by the Transform component.
    product_vocab = tf_transform_output.vocabulary_by_name("vocabulary_product_id")
    product_lookup_layer = tf.keras.layers.StringLookup(
        vocabulary=product_vocab, num_oov_indices=1
    )

    def preprocess_product_features(features):
        """Applies the product_id transformation."""
        # The input `features` is a dictionary like {"product_id": "some_id_string"}
        transformed_id = product_lookup_layer(features["product_id"])
        return {transformed_name("product_id"): transformed_id}

    # Map the raw product IDs to their embeddings to create the candidate set for metrics.
    candidates = products_dict_ds.batch(4096).map(preprocess_product_features).map(
        lambda x: model.product_model(x[transformed_name("product_id")])
    )

    model.task.factorized_metrics = tfrs.metrics.FactorizedTopK(candidates=candidates)

    model.fit(
        train_dataset,
        epochs=fn_args.custom_config.get("epochs", 5),
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback]
    )

    # Create and save retrieval index
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.index_from_dataset(
        dataset=products_dict_ds.batch(4096).map(
            lambda x: (
                x["product_id"],  # The raw string ID for retrieval
                model.product_model(preprocess_product_features(x)[transformed_name("product_id")])
            )
        )
    )

    # Define and save serving model
    @tf.function
    def serving_fn(serialized_tf_examples):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        serving_feature_spec = {
            # We need all raw features that are inputs to the user model's
            # branch of the transform graph.
            "cust_person_id": raw_feature_spec["cust_person_id"],
            "city": raw_feature_spec["city"],
            "date_of_day": raw_feature_spec["date_of_day"],
            "total_revenue": raw_feature_spec["total_revenue"],
        }
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, serving_feature_spec
        )
        transformed_features = tft_layer(parsed_features)
        # The user model expects a dictionary of transformed features.
        user_embeddings = index._query_model(transformed_features)
        _, titles = index(user_embeddings)
        return {"product_id": titles}

    concrete_serving_fn = serving_fn.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
    )
    signatures = {"serving_default": concrete_serving_fn}
    tf.saved_model.save(index, fn_args.serving_model_dir, signatures=signatures)
