"""Custom BigQuery ExampleGen component for large-scale data processing."""

import os
from typing import Optional, Dict, Any
import apache_beam as beam

# Set protobuf implementation to avoid version conflicts
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from tfx.components.example_gen import base_example_gen_executor
from tfx.dsl.components.base import executor_spec
from tfx.proto import example_gen_pb2

# Use TFX's BigQuery extension
from tfx.extensions.google_cloud_big_query.example_gen.component import (
    BigQueryExampleGen,
)


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(bytes)
def _BigQueryToExample(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline, exec_properties: Dict[str, Any], split_pattern: str
) -> beam.pvalue.PCollection:
    """Reads from BigQuery in a scalable way and creates tf.Examples."""
    query = exec_properties["custom_config"]["query"]
    project_id = exec_properties["custom_config"]["project_id"]

    def format_row(bq_row: Dict[str, Any]) -> Dict[str, Any]:
        """Formats a BigQuery row into a dictionary with the correct feature names and types."""
        # Ensure all values are properly typed for TFX processing
        formatted_row = {
            "cust_person_id": str(bq_row.get("cust_person_id", "")),
            "product_id": str(bq_row.get("product_id", "")),
            "city": str(bq_row.get("city", "")),
            "sell_val_nsp": float(bq_row.get("sell_val_nsp", 0.0)),
            "date_of_day": str(bq_row.get("date_of_day", "")),
            "total_revenue": float(bq_row.get("total_revenue", 0.0)),
        }
        return formatted_row

    def row_to_example(formatted_row: Dict[str, Any]) -> bytes:
        """Convert formatted row to serialized TF Example."""
        # Ensure protobuf compatibility
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

        # Use TensorFlow to create the example
        import tensorflow as tf

        feature_dict = {}

        # Add string features
        for key in ["cust_person_id", "product_id", "city", "date_of_day"]:
            value = formatted_row.get(key, "")
            feature_dict[key] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[value.encode("utf-8")])
            )

        # Add float features
        for key in ["sell_val_nsp", "total_revenue"]:
            value = float(formatted_row.get(key, 0.0))
            feature_dict[key] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[value])
            )

        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return example.SerializeToString()

    return (
        pipeline
        | "ReadFromBigQuery"
        >> beam.io.ReadFromBigQuery(
            query=query,
            use_standard_sql=True,
            project=project_id,
            # Simplified BigQuery read configuration
            use_json_exports=True,  # Use JSON exports for better compatibility
        )
        | "FormatRow" >> beam.Map(format_row)
        | "ToTFExample" >> beam.Map(row_to_example)
    )


class BigQueryExampleGenExecutor(base_example_gen_executor.BaseExampleGenExecutor):
    """Custom executor for BigQuery ExampleGen optimized for large-scale data."""

    def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
        """Returns PTransform for BigQuery to TF examples."""
        return _BigQueryToExample


def create_bigquery_example_gen(
    query: str,
    project_id: str,
    output_config: Optional[example_gen_pb2.Output] = None,
    beam_pipeline_args: Optional[list] = None,
) -> BigQueryExampleGen:
    """Creates a BigQuery ExampleGen component optimized for large datasets."""

    # Simplified beam args for better compatibility
    default_beam_args = [
        "--runner=DataflowRunner",
        f"--project={project_id}",
        "--region=europe-west4",
        "--temp_location=gs://recs_metroua/dataflow_temp",
        "--staging_location=gs://recs_metroua/dataflow_staging",
        "--num_workers=2",
        "--max_num_workers=10",
        "--worker_machine_type=n1-standard-2",
        "--disk_size_gb=50",
        "--no_use_public_ips",
        # Essential experiments only
        "--experiments=use_runner_v2",
        "--save_main_session",
    ]

    # Merge with provided beam args
    final_beam_args = default_beam_args
    if beam_pipeline_args:
        # Override defaults with provided args
        provided_keys = [arg.split("=")[0] for arg in beam_pipeline_args if "=" in arg]
        final_beam_args = [
            arg
            for arg in default_beam_args
            if not any(arg.startswith(key) for key in provided_keys)
        ]
        final_beam_args.extend(beam_pipeline_args)

    # Use standard BigQueryExampleGen instead of custom executor
    # This avoids the import path issues with custom executors
    component = BigQueryExampleGen(
        query=query,
        output_config=output_config,
    )

    # Apply beam pipeline args using the with_beam_pipeline_args method
    component = component.with_beam_pipeline_args(final_beam_args)

    return component


# Make the executor class available at module level for TFX import
__all__ = ["BigQueryExampleGenExecutor", "create_bigquery_example_gen"]
        output_config=output_config,
    )

    # Apply beam pipeline args using the with_beam_pipeline_args method
    component = component.with_beam_pipeline_args(final_beam_args)

    return component


# Make the executor class available at module level for TFX import
__all__ = ["BigQueryExampleGenExecutor", "create_bigquery_example_gen"]
