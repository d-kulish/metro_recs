"""Custom BigQuery ExampleGen component."""

from typing import Optional, Dict, Any
import apache_beam as beam
import tensorflow as tf
from google.cloud import bigquery
import pandas as pd

from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor
from tfx.components.example_gen.component import FileBasedExampleGen
from tfx.components.example_gen import utils
from tfx.dsl.components.base import executor_spec
from tfx.types import standard_artifacts


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _BigQueryToExample(
    pipeline: beam.Pipeline, exec_properties: Dict[str, Any], split_pattern: str
) -> beam.pvalue.PCollection:
    """Read from BigQuery and create tf.Examples."""

    def query_to_dict():
        """Query BigQuery and yield dictionaries."""
        project_id = exec_properties["custom_config"]["project_id"]
        query = exec_properties["custom_config"]["query"]

        client = bigquery.Client(project=project_id)
        df = client.query(query).to_dataframe()

        for _, row in df.iterrows():
            yield {
                "user_id": str(row["cust_person_id"]),
                "product_id": str(row["product_id"]),
                "city": str(row["city"]),
                "sell_val_nsp": float(row["sell_val_nsp"]),
                "date_of_day": str(row["date_of_day"]),
            }

    return (
        pipeline
        | "CreateData" >> beam.Create(list(query_to_dict()))
        | "ToTFExample" >> beam.Map(utils.dict_to_example)
    )


class BigQueryExampleGenExecutor(BaseExampleGenExecutor):
    """Custom executor for BigQuery ExampleGen."""

    def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
        """Returns PTransform for BigQuery to TF examples."""
        return _BigQueryToExample


def create_bigquery_example_gen(
    query: str,
    project_id: str,
    output_config: Optional[standard_artifacts.ExampleGen.OutputConfig] = None,
) -> FileBasedExampleGen:
    """Creates a BigQuery ExampleGen component.

    Args:
        query: SQL query to execute
        project_id: GCP project ID
        output_config: Optional output configuration

    Returns:
        FileBasedExampleGen component
    """
    return FileBasedExampleGen(
        input_base="dummy",  # Not used for BigQuery
        custom_config={"query": query, "project_id": project_id},
        output_config=output_config,
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            BigQueryExampleGenExecutor
        ),
    )
