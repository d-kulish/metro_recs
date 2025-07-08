"""Trainer module for Metro recommendations model."""

from typing import Dict, List, Text
import os
import glob
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_recommenders as tfrs
from absl import logging

from tfx.types import artifact_utils
from tfx import v1 as tfx
from tfx_bsl.coders import example_coder
from tfx_bsl.public import tfxio

from pipeline.modules.transform_module import transformed_name, LABEL_KEY

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

    def __init__(self, user_model, product_model):
        super().__init__()
        self.product_model = product_model
        self.user_model = user_model

        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK()
        )

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        """Compute loss for training."""
        # We need to handle the extra dimension added by TFT.
        user_embeddings = self.user_model(features[transformed_name("cust_person_id")])
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
    """Builds the user model tower."""
    vocab_size = tf_transform_output.vocabulary_size_by_name("vocabulary_cust_person_id")
    user_id_input = tf.keras.Input(
        shape=(1,), name=transformed_name("cust_person_id"), dtype=tf.int64
    )
    embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=EMBEDDING_DIMENSION
    )(user_id_input)
    return tf.keras.Model(inputs=user_id_input, outputs=tf.keras.layers.Flatten()(embedding))


def _build_product_model(
    tf_transform_output: tft.TFTransformOutput,
) -> tf.keras.Model:
    """Builds the product model tower."""
    vocab_size = tf_transform_output.vocabulary_size_by_name("vocabulary_product_id")
    product_id_input = tf.keras.Input(
        shape=(1,), name=transformed_name("product_id"), dtype=tf.int64
    )
    embedding = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=EMBEDDING_DIMENSION
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

    # Set up the candidate dataset for metrics
    products_artifact = fn_args.custom_config["products"].get()[0]
    input_dir = artifact_utils.get_split_uri([products_artifact], "train")
    product_files = glob.glob(os.path.join(input_dir, "*"))
    products_ds = tf.data.TFRecordDataset(product_files, compression_type="GZIP")

    def parse_product_example(proto):
        return tf.io.parse_single_example(
            proto, {"product_id": tf.io.FixedLenFeature([], tf.string)}
        )

    candidates = products_ds.map(parse_product_example).map(tft_layer).map(
        lambda x: x[transformed_name("product_id")]
    ).batch(4096).map(model.product_model)

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
        dataset=products_ds.map(parse_product_example).batch(4096).map(
            lambda x: (
                x["product_id"],
                model.product_model(tft_layer(x)[transformed_name("product_id")]),
            )
        )
    )

    # Define and save serving model
    @tf.function
    def serving_fn(serialized_tf_examples):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        serving_feature_spec = {
            "cust_person_id": raw_feature_spec["cust_person_id"]
        }
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, serving_feature_spec
        )
        transformed_features = tft_layer(parsed_features)
        user_embeddings = index._query_model(
            transformed_features[transformed_name("cust_person_id")]
        )
        _, titles = index(user_embeddings)
        return {"product_id": titles}

    concrete_serving_fn = serving_fn.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
    )
    signatures = {"serving_default": concrete_serving_fn}
    tf.saved_model.save(index, fn_args.serving_model_dir, signatures=signatures)
