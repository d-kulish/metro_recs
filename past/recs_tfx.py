# Setting up 
## Installing libraries
!pip install -Uq tfx
!pip install -Uq tensorflow-recommenders
!pip install -Uq tensorflow-datasets

## Loading libraries 
import os
import absl
import json
import pprint
import tempfile

from typing import Any, Dict, List, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import apache_beam as beam

from absl import logging

from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor
from tfx.components.example_gen.component import FileBasedExampleGen
from tfx.components.example_gen import utils
from tfx.dsl.components.base import executor_spec

from tfx.types import artifact
from tfx.types import artifact_utils
from tfx.types import channel
from tfx.types import standard_artifacts
from tfx.types.standard_artifacts import Examples

from tfx.dsl.component.experimental.annotations import InputArtifact
from tfx.dsl.component.experimental.annotations import OutputArtifact
from tfx.dsl.component.experimental.annotations import Parameter
from tfx.dsl.component.experimental.decorators import component
from tfx.types.experimental.simple_artifacts import Dataset

from tfx import v1 as tfx
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

# Set up logging.
tf.get_logger().propagate = False
absl.logging.set_verbosity(absl.logging.INFO)
pp = pprint.PrettyPrinter()

print(f"TensorFlow version: {tf.__version__}")
print(f"TFX version: {tfx.__version__}")
print(f"TensorFlow Recommenders version: {tfrs.__version__}")

# Example Gen Component
@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _TFDatasetToExample(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline,
    exec_properties: Dict[str, Any],
    split_pattern: str
    ) -> beam.pvalue.PCollection:
    """Read a TensorFlow Dataset and create tf.Examples"""
    custom_config = json.loads(exec_properties['custom_config'])
    dataset_name = custom_config['dataset']
    split_name = custom_config['split']

    builder = tfds.builder(dataset_name)
    builder.download_and_prepare()

    return (pipeline
            | 'MakeExamples' >> tfds.beam.ReadFromTFDS(builder, split=split_name)
            | 'AsNumpy' >> beam.Map(tfds.as_numpy)
            | 'ToDict' >> beam.Map(dict)
            | 'ToTFExample' >> beam.Map(utils.dict_to_example)
            )

class TFDSExecutor(BaseExampleGenExecutor):
  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for TF Dataset to TF examples."""
    return _TFDatasetToExample
  
context = InteractiveContext()

# Preparing the dataset
ratings_example_gen = FileBasedExampleGen(
    input_base='dummy',
    custom_config={'dataset':'movielens/100k-ratings', 'split':'train'},
    custom_executor_spec=executor_spec.ExecutorClassSpec(TFDSExecutor))
context.run(ratings_example_gen, enable_cache=True)

movies_example_gen = FileBasedExampleGen(
    input_base='dummy',
    custom_config={'dataset':'movielens/100k-movies', 'split':'train'},
    custom_executor_spec=executor_spec.ExecutorClassSpec(TFDSExecutor))
context.run(movies_example_gen, enable_cache=True)

# Create `inspect_examples` utility - not sure it is needed 
def inspect_examples(component,
                     channel_name='examples',
                     split_name='train',
                     num_examples=1):
  # Get the URI of the output artifact, which is a directory
  full_split_name = 'Split-{}'.format(split_name)
  print('channel_name: {}, split_name: {} (\"{}\"), num_examples: {}\n'.format(
      channel_name, split_name, full_split_name, num_examples))
  train_uri = os.path.join(
      component.outputs[channel_name].get()[0].uri, full_split_name)

  # Get the list of files in this directory (all compressed TFRecord files)
  tfrecord_filenames = [os.path.join(train_uri, name)
                        for name in os.listdir(train_uri)]

  # Create a `TFRecordDataset` to read these files
  dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")

  # Iterate over the records and print them
  for tfrecord in dataset.take(num_examples):
    serialized_example = tfrecord.numpy()
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    pp.pprint(example)

inspect_examples(ratings_example_gen)

# Generate statistics for movies and ratings
movies_stats_gen = tfx.components.StatisticsGen(
    examples=movies_example_gen.outputs['examples'])
context.run(movies_stats_gen, enable_cache=True)

ratings_stats_gen = tfx.components.StatisticsGen(
    examples=ratings_example_gen.outputs['examples'])
context.run(ratings_stats_gen, enable_cache=True)

# Create schema for movies and ratings
movies_schema_gen = tfx.components.SchemaGen(
    statistics=movies_stats_gen.outputs['statistics'],
    infer_feature_shape=False)
context.run(movies_schema_gen, enable_cache=True)

ratings_schema_gen = tfx.components.SchemaGen(
    statistics=ratings_stats_gen.outputs['statistics'],
    infer_feature_shape=False)
context.run(ratings_schema_gen, enable_cache=True)

# Feature Engineering using Transform
_movies_transform_module_file = 'movies_transform_module.py'
%%writefile {_movies_transform_module_file}

import tensorflow as tf
import tensorflow_transform as tft

def preprocessing_fn(inputs):
  # We only want the movie title
  return {'movie_title':inputs['movie_title']}

movies_transform = tfx.components.Transform(
    examples=movies_example_gen.outputs['examples'],
    schema=movies_schema_gen.outputs['schema'],
    module_file=os.path.abspath(_movies_transform_module_file))
context.run(movies_transform, enable_cache=True)

_ratings_transform_module_file = 'ratings_transform_module.py'
%%writefile {_ratings_transform_module_file}

import tensorflow as tf
import tensorflow_transform as tft
import pdb

NUM_OOV_BUCKETS = 1

def preprocessing_fn(inputs):
  # We only want the user ID and the movie title, but we also need vocabularies
  # for both of them.  The vocabularies aren't features, they're only used by
  # the lookup.
  outputs = {}
  outputs['user_id'] = tft.sparse_tensor_to_dense_with_shape(inputs['user_id'], [None, 1], '-1')
  outputs['movie_title'] = tft.sparse_tensor_to_dense_with_shape(inputs['movie_title'], [None, 1], '-1')

  tft.compute_and_apply_vocabulary(
      inputs['user_id'],
      num_oov_buckets=NUM_OOV_BUCKETS,
      vocab_filename='user_id_vocab')

  tft.compute_and_apply_vocabulary(
      inputs['movie_title'],
      num_oov_buckets=NUM_OOV_BUCKETS,
      vocab_filename='movie_title_vocab')

  return outputs

ratings_transform = tfx.components.Transform(
    examples=ratings_example_gen.outputs['examples'],
    schema=ratings_schema_gen.outputs['schema'],
    module_file=os.path.abspath(_ratings_transform_module_file))
context.run(ratings_transform, enable_cache=True)

# Implementing a model in TFX 

_trainer_module_file = 'trainer_module.py'

%%writefile {_trainer_module_file}

from typing import Dict, List, Text

import pdb

import os
import absl
import datetime
import glob
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_recommenders as tfrs

from absl import logging
from tfx.types import artifact_utils

from tfx import v1 as tfx
from tfx_bsl.coders import example_coder
from tfx_bsl.public import tfxio

absl.logging.set_verbosity(absl.logging.INFO)

EMBEDDING_DIMENSION = 32
INPUT_FN_BATCH_SIZE = 1


def extract_str_feature(dataset, feature_name):
  np_dataset = []
  for example in dataset:
    np_example = example_coder.ExampleToNumpyDict(example.numpy())
    np_dataset.append(np_example[feature_name][0].decode())
  return tf.data.Dataset.from_tensor_slices(np_dataset)


class MovielensModel(tfrs.Model):

  def __init__(self, user_model, movie_model, tf_transform_output, movies_uri):
    super().__init__()
    self.movie_model: tf.keras.Model = movie_model
    self.user_model: tf.keras.Model = user_model

    movies_artifact = movies_uri.get()[0]
    input_dir = artifact_utils.get_split_uri([movies_artifact], 'train')
    movie_files = glob.glob(os.path.join(input_dir, '*'))
    movies = tf.data.TFRecordDataset(movie_files, compression_type="GZIP")
    movies_dataset = extract_str_feature(movies, 'movie_title')

    loss_metrics = tfrs.metrics.FactorizedTopK(
        candidates=movies_dataset.batch(128).map(movie_model)
        )

    self.task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=loss_metrics
        )


  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    try:
      user_embeddings = tf.squeeze(self.user_model(features['user_id']), axis=1)
      # And pick out the movie features and pass them into the movie model,
      # getting embeddings back.
      positive_movie_embeddings = self.movie_model(features['movie_title'])

      # The task computes the loss and the metrics.
      _task = self.task(user_embeddings, positive_movie_embeddings)
    except BaseException as err:
      logging.error('######## ERROR IN compute_loss:\n{}\n###############'.format(err))

    return _task


# This function will apply the same transform operation to training data
# and serving requests.
def _apply_preprocessing(raw_features, tft_layer):
  try:
    transformed_features = tft_layer(raw_features)
  except BaseException as err:
    logging.error('######## ERROR IN _apply_preprocessing:\n{}\n###############'.format(err))

  return transformed_features


def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  try:
    return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size),
      tf_transform_output.transformed_metadata.schema)
  except BaseException as err:
    logging.error('######## ERROR IN _input_fn:\n{}\n###############'.format(err))

  return None


def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example and applies TFT."""
  try:
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
      """Returns the output to be used in the serving signature."""
      try:
        feature_spec = tf_transform_output.raw_feature_spec()
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        result = model(transformed_features)
      except BaseException as err:
        logging.error('######## ERROR IN serve_tf_examples_fn:\n{}\n###############'.format(err))
      return result
  except BaseException as err:
      logging.error('######## ERROR IN _get_serve_tf_examples_fn:\n{}\n###############'.format(err))

  return serve_tf_examples_fn


def _build_user_model(
    tf_transform_output: tft.TFTransformOutput, # Specific to ratings
    embedding_dimension: int = 32) -> tf.keras.Model:
  """Creates a Keras model for the query tower.

  Args:
    tf_transform_output: [tft.TFTransformOutput], the results of Transform
    embedding_dimension: [int], the dimensionality of the embedding space

  Returns:
    A keras Model.
  """
  try:
    unique_user_ids = tf_transform_output.vocabulary_by_name('user_id_vocab')
    users_vocab_str = [b.decode() for b in unique_user_ids]

    model = tf.keras.Sequential(
        [
         tf.keras.layers.StringLookup(
             vocabulary=users_vocab_str, mask_token=None),
         # We add an additional embedding to account for unknown tokens.
         tf.keras.layers.Embedding(len(users_vocab_str) + 1, embedding_dimension)
         ])
  except BaseException as err:
    logging.error('######## ERROR IN _build_user_model:\n{}\n###############'.format(err))

  return model


def _build_movie_model(
    tf_transform_output: tft.TFTransformOutput, # Specific to movies
    embedding_dimension: int = 32) -> tf.keras.Model:
  """Creates a Keras model for the candidate tower.

  Args:
    tf_transform_output: [tft.TFTransformOutput], the results of Transform
    embedding_dimension: [int], the dimensionality of the embedding space

  Returns:
    A keras Model.
  """
  try:
    unique_movie_titles = tf_transform_output.vocabulary_by_name('movie_title_vocab')
    titles_vocab_str = [b.decode() for b in unique_movie_titles]

    model = tf.keras.Sequential(
        [
         tf.keras.layers.StringLookup(
             vocabulary=titles_vocab_str, mask_token=None),
         # We add an additional embedding to account for unknown tokens.
         tf.keras.layers.Embedding(len(titles_vocab_str) + 1, embedding_dimension)
        ])
  except BaseException as err:
      logging.error('######## ERROR IN _build_movie_model:\n{}\n###############'.format(err))
  return model


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  try:
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, fn_args.data_accessor,
                              tf_transform_output, INPUT_FN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, fn_args.data_accessor,
                            tf_transform_output, INPUT_FN_BATCH_SIZE)

    model = MovielensModel(
        _build_user_model(tf_transform_output, EMBEDDING_DIMENSION),
        _build_movie_model(tf_transform_output, EMBEDDING_DIMENSION),
        tf_transform_output,
        fn_args.custom_config['movies']
        )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch')

    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
  except BaseException as err:
    logging.error('######## ERROR IN run_fn before fit:\n{}\n###############'.format(err))

  try:
    model.fit(
        train_dataset,
        epochs=fn_args.custom_config['epochs'],
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback])
  except BaseException as err:
      logging.error('######## ERROR IN run_fn during fit:\n{}\n###############'.format(err))

  try:
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

    movies_artifact = fn_args.custom_config['movies'].get()[0]
    input_dir = artifact_utils.get_split_uri([movies_artifact], 'eval')
    movie_files = glob.glob(os.path.join(input_dir, '*'))
    movies = tf.data.TFRecordDataset(movie_files, compression_type="GZIP")

    movies_dataset = extract_str_feature(movies, 'movie_title')

    index.index_from_dataset(
      tf.data.Dataset.zip((
          movies_dataset.batch(100),
          movies_dataset.batch(100).map(model.movie_model))
      )
    )

    # Run once so that we can get the right signatures into SavedModel
    _, titles = index(tf.constant(["42"]))
    print(f"Recommendations for user 42: {titles[0, :3]}")

    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(index,
                                      tf_transform_output).get_concrete_function(
                                          tf.TensorSpec(
                                              shape=[None],
                                              dtype=tf.string,
                                              name='examples')),
    }
    index.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)

  except BaseException as err:
      logging.error('######## ERROR IN run_fn during export:\n{}\n###############'.format(err))

# Training Model 
trainer = tfx.components.Trainer(
    module_file=os.path.abspath(_trainer_module_file),
    examples=ratings_transform.outputs['transformed_examples'],
    transform_graph=ratings_transform.outputs['transform_graph'],
    schema=ratings_transform.outputs['post_transform_schema'],
    train_args=tfx.proto.TrainArgs(num_steps=500),
    eval_args=tfx.proto.EvalArgs(num_steps=10),
    custom_config={
        'epochs':5,
        'movies':movies_transform.outputs['transformed_examples'],
        'movie_schema':movies_transform.outputs['post_transform_schema'],
        'ratings':ratings_transform.outputs['transformed_examples'],
        'ratings_schema':ratings_transform.outputs['post_transform_schema']
        })

context.run(trainer, enable_cache=False)

# Exporting the model
_serving_model_dir = os.path.join(tempfile.mkdtemp(), 'serving_model/tfrs_retrieval')

pusher = tfx.components.Pusher(
    model=trainer.outputs['model'],
    push_destination=tfx.proto.PushDestination(
        filesystem=tfx.proto.PushDestination.Filesystem(
            base_directory=_serving_model_dir)))
context.run(pusher, enable_cache=True)
