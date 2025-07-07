"""Trainer module for Metro recommendations model."""

from typing import Dict, Text
import os
import glob
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_recommenders as tfrs
from absl import logging
from tfx.types import artifact_utils
from tfx import v1 as tfx
from tfx_bsl.coders import example_coder

EMBEDDING_DIMENSION = 32


def extract_str_feature(dataset, feature_name):
    """Extract string feature from dataset."""
    np_dataset = []
    for example in dataset:
        np_example = example_coder.ExampleToNumpyDict(example.numpy())
        np_dataset.append(np_example[feature_name][0].decode())
    return tf.data.Dataset.from_tensor_slices(np_dataset)


class MetroRecommendationModel(tfrs.Model):
    """Metro recommendation model using TFX."""

    def __init__(self, user_model, product_model, tf_transform_output, products_uri):
        super().__init__()
        self.product_model = product_model
        self.user_model = user_model

        # Load products for candidate selection
        products_artifact = products_uri.get()[0]
        input_dir = artifact_utils.get_split_uri([products_artifact], "train")
        product_files = glob.glob(os.path.join(input_dir, "*"))
        products = tf.data.TFRecordDataset(product_files, compression_type="GZIP")
        products_dataset = extract_str_feature(products, "product_id")

        loss_metrics = tfrs.metrics.FactorizedTopK(
            candidates=products_dataset.batch(128).map(product_model)
        )

        self.task = tfrs.tasks.Retrieval(metrics=loss_metrics)

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        """Compute loss for training."""
        try:
            user_embeddings = tf.squeeze(self.user_model(features["user_id"]), axis=1)
            positive_product_embeddings = self.product_model(features["product_id"])
            return self.task(user_embeddings, positive_product_embeddings)
        except Exception as err:
            logging.error(f"Error in compute_loss: {err}")
            raise


def _build_user_model(
    tf_transform_output: tft.TFTransformOutput, embedding_dimension: int = 32
) -> tf.keras.Model:
    """Build user embedding model."""
    unique_user_ids = tf_transform_output.vocabulary_by_name("user_id_vocab")
    users_vocab_str = [b.decode() for b in unique_user_ids]

    return tf.keras.Sequential(
        [
            tf.keras.layers.StringLookup(vocabulary=users_vocab_str, mask_token=None),
            tf.keras.layers.Embedding(len(users_vocab_str) + 1, embedding_dimension),
        ]
    )


def _build_product_model(
    tf_transform_output: tft.TFTransformOutput, embedding_dimension: int = 32
) -> tf.keras.Model:
    """Build product embedding model."""
    unique_product_ids = tf_transform_output.vocabulary_by_name("product_id_vocab")
    products_vocab_str = [b.decode() for b in unique_product_ids]

    return tf.keras.Sequential(
        [
            tf.keras.layers.StringLookup(
                vocabulary=products_vocab_str, mask_token=None
            ),
            tf.keras.layers.Embedding(len(products_vocab_str) + 1, embedding_dimension),
        ]
    )


def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=200):
    """Create input function for training."""
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfx.components.DataAccessor.TensorFlowDatasetOptions(batch_size=batch_size),
        tf_transform_output.transformed_metadata.schema,
    )


def run_fn(fn_args: tfx.components.FnArgs):
    """Main training function called by TFX Trainer."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Create datasets
    train_dataset = _input_fn(
        fn_args.train_files, fn_args.data_accessor, tf_transform_output, 32
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, fn_args.data_accessor, tf_transform_output, 32
    )

    # Build model
    model = MetroRecommendationModel(
        _build_user_model(tf_transform_output, EMBEDDING_DIMENSION),
        _build_product_model(tf_transform_output, EMBEDDING_DIMENSION),
        tf_transform_output,
        fn_args.custom_config["products"],
    )

    # Compile and train
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq="batch"
    )

    model.fit(
        train_dataset,
        epochs=fn_args.custom_config.get("epochs", 5),
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
    )

    # Create and save retrieval index
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

    # Index products
    products_artifact = fn_args.custom_config["products"].get()[0]
    input_dir = artifact_utils.get_split_uri([products_artifact], "eval")
    product_files = glob.glob(os.path.join(input_dir, "*"))
    products = tf.data.TFRecordDataset(product_files, compression_type="GZIP")
    products_dataset = extract_str_feature(products, "product_id")

    index.index_from_dataset(
        tf.data.Dataset.zip(
            (
                products_dataset.batch(100),
                products_dataset.batch(100).map(model.product_model),
            )
        )
    )

    # Save model
    index.save(fn_args.serving_model_dir, save_format="tf")
