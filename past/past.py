# Setting up 
## Installing specifi libraries 
!pip install tensorflow-recommenders --no-deps

!pip install tensorflow==2.11.0

## Loading libraries 
import pandas as pd
import tensorflow as tf
import os
import glob
import numpy as np
import tensorflow_recommenders as tfrs
import matplotlib.pyplot as plt
from typing import Dict, Text
import datetime as dt
import pickle
import random

print(tfrs.__version__)
print(tf.__version__)

# Loading data 
## Loading dataframe 
result = pd.read_csv('/content/drive/MyDrive/Work/C4R/Metro/Data/ml_300.csv')
## Converting data to the correct dtypes 
result[['cust_person_id', 'product_id']] = result[['cust_person_id', 'product_id']].astype(str)
result[['visits', 'sell_total_val_nsp', 'sell_val_nsp']] = result[['visits', 'sell_total_val_nsp', 'sell_val_nsp']].fillna(0)
result[['visits', 'sell_total_val_nsp', 'sell_val_nsp']] = result[['visits', 'sell_total_val_nsp', 'sell_val_nsp']].astype(int)
result[['SEGMENT_GROUP']] = result[['SEGMENT_GROUP']].fillna('unknown')

# Select top products 
## Calculate cumulative percentage of sales for each product
product_df = result.groupby('product_id')['sell_val_nsp'].sum().reset_index()
product_df = product_df.sort_values(by='sell_val_nsp', ascending=False)
product_df['cumulative_percentage'] = (product_df['sell_val_nsp'].cumsum() / product_df['sell_val_nsp'].sum()) * 100
## Select top products based on cumulative percentage
temp_df = product_df[(product_df['cumulative_percentage'] <= 80)]
## Create a set of top products
product_list = list(set(temp_df['product_id'].to_list()))
## Create final df with top products
filtered_df = result[result['product_id'].isin(product_list)]

# Creating TF records 
## Function to save DataFrame to TFRecords
def save_to_tf_records(df, output_dir, rows_per_file=1000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(0, len(df), rows_per_file):
        chunk = df[i:i+rows_per_file]
        filename = f"data_{i:05d}.tfrecord"
        filepath = os.path.join(output_dir, filename)
        with tf.io.TFRecordWriter(filepath) as writer:
            for _, row in chunk.iterrows():
                example = tf.train.Example(features=tf.train.Features(feature={
                    'customer_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row["cust_person_id"].encode()])),
                    'product_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row["product_id"].encode()])),
                    # 'segment': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row["SEGMENT_GROUP"].encode()])),
                    # "visits": tf.train.Feature(int64_list=tf.train.Int64List(value=[row["visits"]])),
                    "revenue": tf.train.Feature(int64_list=tf.train.Int64List(value=[row["sell_total_val_nsp"]])),
                    'city': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row["city"].encode()])),
                    # 'articul': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row["art_name_tl"].encode()])),
                    # 'category': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row["pcg_cat_desc_tl"].encode()])),
                    # 'sub_category': tf.train.Feature(bytes_list=tf.train.BytesList(value=[row["pcg_sub_cat_desc_tl"].encode()])),
                    # "ts": tf.train.Feature(int64_list=tf.train.Int64List(value=[row["ts"]])),
                    # 'rating': tf.train.Feature(int64_list=tf.train.Int64List(value=[row["sell_val_nsp"]])),
                    }))

                writer.write(example.SerializeToString())
## Save DataFrame to TFRecords files 
save_to_tf_records(filtered_df, "tf_recs", rows_per_file=200_000)

# Creating TF dataset 
## Create full datase 
feature_description = {
    'product_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'customer_id': tf.io.FixedLenFeature([], tf.string, default_value=''),
    # 'segment': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'city': tf.io.FixedLenFeature([], tf.string, default_value=''),
    # 'articul': tf.io.FixedLenFeature([], tf.string, default_value=''),
    # 'category': tf.io.FixedLenFeature([], tf.string, default_value=''),
    # 'sub_category': tf.io.FixedLenFeature([], tf.string, default_value=''),
    # 'visits': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'revenue': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    # 'ts': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    # 'rating': tf.io.FixedLenFeature([], tf.int64, default_value=0),
 }

def _parse_function(example_proto):
  return tf.io.parse_single_example(example_proto, feature_description)

tfrecord_files = glob.glob('tf_recs/*')

full_ds = tf.data.TFRecordDataset(tfrecord_files).map(_parse_function)

##  Create Client dataset
client_ds = full_ds.map(lambda x: {
    "product_id": x["product_id"],
    'customer_id': x['customer_id'],
    # "segment": x['segment'],
    'city': x['city'],
    # 'visits': x['visits'],
    'revenue': x['revenue'],
    # 'category': x['category'],
    # 'sub_category': x['sub_category'],
    # 'articul': x['articul'],
    # 'ts': x['ts'],
    # 'rating': x['rating'],
})

## Create Product dataset
unique_product_ids_df = pd.DataFrame(filtered_df['product_id'].unique(), columns=['product_id'])
product_ids = {key: col.values for key, col in dict(unique_product_ids_df[['product_id']]).items()}
product_ds = tf.data.Dataset.from_tensor_slices(product_ids)

# Creating Vocabs 
## Unique revenues 
revenues = np.concatenate(list(client_ds.map(lambda x: x["revenue"]).batch(1_000)))
max_revenues = revenues.max()
min_revenues = revenues.min()

revenues_buckets = np.linspace(
    min_revenues, max_revenues, num=100,
)
unique_revenues = np.unique(np.concatenate(list(client_ds.batch(1_000).map(
                                                  lambda x: x['revenue']))))

## Unique products 
unique_product_ids = np.unique(np.concatenate(list(product_ds.batch(1_000).map(
    lambda x: x["product_id"]))))

## Unique customer IDs 
unique_customer_id = np.unique(np.concatenate(list(client_ds.batch(1_000).map(
    lambda x: x["customer_id"]))))

## Unique cities
unique_city = np.unique(np.concatenate(list(client_ds.batch(1_000).map(
    lambda x: x["city"]))))

# Model creation
## Buyer Model 
class BuyerModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    self.customers_embedding = tf.keras.Sequential([
       tf.keras.layers.StringLookup(
         vocabulary = unique_customer_id, mask_token = None),
         tf.keras.layers.Embedding(len(unique_customer_id) + 1, 32),
     ])

    self.city_embedding = tf.keras.Sequential([
       tf.keras.layers.StringLookup(
         vocabulary = unique_city, mask_token = None),
         tf.keras.layers.Embedding(len(unique_city) + 1, 32),
     ])

    # self.category_embedding = tf.keras.Sequential([
    #    tf.keras.layers.StringLookup(
    #      vocabulary = unique_category, mask_token = None),
    #      tf.keras.layers.Embedding(len(unique_category) + 1, 32),
    #  ])

    self.revenues_embedding = tf.keras.Sequential([
          tf.keras.layers.Discretization(revenues_buckets.tolist()),
          tf.keras.layers.Embedding(len(revenues_buckets) + 1, 32),])
    self.normalized_revenues = tf.keras.layers.Normalization(axis=None)
    self.normalized_revenues.adapt(unique_revenues)

    # self.visits_embedding = tf.keras.Sequential([
    #       tf.keras.layers.Discretization(visits_buckets.tolist()),
    #       tf.keras.layers.Embedding(len(visits_buckets) + 1, 32),])
    # self.normalized_visits = tf.keras.layers.Normalization(axis=None)
    # self.normalized_visits.adapt(unique_visits)

  def call(self, inputs):
    return tf.concat([
      self.customers_embedding(inputs['customer_id']),
      self.city_embedding(inputs['city']),
      # self.category_embedding(inputs['category']),
      self.revenues_embedding(inputs["revenue"]),
      tf.reshape(self.normalized_revenues(inputs["revenue"]), (-1, 1)),
      # self.visits_embedding(inputs["visits"]),
      # tf.reshape(self.normalized_visits(inputs["visits"]), (-1, 1)),
      ], axis=1)

buyer_model = BuyerModel()

## Product Model
class ProductModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    self.product_embedding = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_product_ids, mask_token=None),
        tf.keras.layers.Embedding(len(unique_product_ids) + 1, 32),
    ])

  def call(self, inputs):
    return tf.concat([
      self.product_embedding(inputs['product_id'])
      ], axis=1)

product_model = ProductModel()

## Joint Model
class Retrival(tfrs.models.Model):

  def __init__(self):
    super().__init__()

    self.query_model = tf.keras.Sequential([
      BuyerModel(),
      tf.keras.layers.Dense(64, kernel_regularizer= tf.keras.regularizers.l1(0.01)),
      tf.keras.layers.Dense(32),
      tf.keras.layers.Dense(16),
      # tf.keras.layers.Dense(8)
       ])

    self.candidate_model = tf.keras.Sequential([
      ProductModel(),
      tf.keras.layers.Dense(64, kernel_regularizer= tf.keras.regularizers.l1(0.01)),
      tf.keras.layers.Dense(32),
      tf.keras.layers.Dense(16),
      # tf.keras.layers.Dense(8)
      ])

    self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=product_ds.batch(128).map(self.candidate_model)
        )
    )

  def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
    buyer_embeddings = self.query_model({
      'customer_id': features['customer_id'],
      # 'category': features['category'],
      'city': features['city'],
      'revenue': features['revenue'],
      # 'visits': features['visits'],
      })

    product_embeddings = self.candidate_model({'product_id': features["product_id"]})

    return buyer_embeddings, product_embeddings


  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    buyer_embeddings, product_embeddings = self(features)

    retrieval_loss = self.retrieval_task(buyer_embeddings, product_embeddings)

    return retrieval_loss

model_retrival = Retrival()

# Training model 
## Splitting, shuffling and batching the dataset
tf.random.set_seed(158)
shuffled = client_ds.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(7_500_000)
test = shuffled.skip(7_500_000)

cached_train = train.batch(10_000).cache()
cached_test = test.batch(5_000).cache()

## Training 
MAX_EPOCHS = 50

def compile_and_fit(model, train_ds, test_ds, patience=3):
   #early_stopping = tf.keras.callbacks.EarlyStopping(monitor='factorized_top_k/top_5_categorical_accuracy',
  #                                                   patience=patience,
  #                                                   mode='min')

  log_dir = "/content/drive/MyDrive/Work/C4R/Metro/logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
  # log_dir = "" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  model.compile(optimizer= tf.keras.optimizers.Adagrad(0.1))

  history = model.fit(train_ds, epochs=MAX_EPOCHS,
                      callbacks = [tensorboard_callback])
  metrics = model.evaluate(test_ds, return_dict=True)

  return model, history, metrics

model, history, metrics = compile_and_fit(model_retrival, cached_train, cached_test)

index = tfrs.layers.factorized_top_k.BruteForce(model.query_model, k = 100)

index.index_from_dataset(
    tf.data.Dataset.zip((product_ds.map(lambda x: x['product_id']).batch(128), product_ds.batch(128).map(model.candidate_model)))
)

score, title = index(buyer_info)

a = np.dstack((title, score))

print(a)

path = os.path.join('/content/drive/MyDrive/Work/C4R/Metro/Models', "UA_P2C_all_sku")

tf.saved_model.save(index, path)
