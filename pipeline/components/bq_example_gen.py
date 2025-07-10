"""Custom BigQuery ExampleGen component."""

from typing import Optional, Dict, Any
import apache_beam as beam

from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor
from tfx.components.example_gen.component import FileBasedExampleGen
from tfx.components.example_gen import utils
from tfx.dsl.components.base import executor_spec
from tfx.types import standard_artifacts

import tensorflow as tf


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _BigQueryToExample(
    pipeline: beam.Pipeline, exec_properties: Dict[str, Any], split_pattern: str
) -> beam.pvalue.PCollection:
    """Reads from BigQuery in a scalable way and creates tf.Examples."""
    project_id = exec_properties["custom_config"]["project_id"]
    query = exec_properties["custom_config"]["query"]

    def format_row(bq_row: Dict[str, Any]) -> Dict[str, Any]:
        """Formats a BigQuery row into a dictionary with the correct feature names and types."""
        return {
            "user_id": str(bq_row["cust_person_id"]),
            "product_id": str(bq_row["product_id"]),
            "city": str(bq_row["city"]),
            "sell_val_nsp": float(bq_row["sell_val_nsp"]),
            "date_of_day": str(bq_row["date_of_day"]),
        }

    return (
        pipeline
        | "ReadFromBigQuery"
        >> beam.io.ReadFromBigQuery(
            query=query, use_standard_sql=True, project=project_id
        )
        | "FormatRow" >> beam.Map(format_row)
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
