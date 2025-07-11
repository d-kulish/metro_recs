"""Custom BigQuery ExampleGen component for large-scale data processing."""

from typing import Optional, Dict, Any
import apache_beam as beam

# Import TF Example protos directly to avoid heavy tensorflow import and recursion errors on Dataflow.
# Correctly import protobuf classes from their respective modules.
from tensorflow.core.example.example_pb2 import Example, Features
from tensorflow.core.example.feature_pb2 import (
    BytesList,
    Feature,
    FloatList,
)

from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor
from tfx.components.example_gen.component import FileBasedExampleGen
from tfx.components.example_gen import utils
from tfx.dsl.components.base import executor_spec
from tfx.types import standard_artifacts


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(Example)
def _BigQueryToExample(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline, exec_properties: Dict[str, Any], split_pattern: str
) -> beam.pvalue.PCollection:
    """Reads from BigQuery in a scalable way and creates tf.Examples."""
    project_id = exec_properties["custom_config"]["project_id"]
    query = exec_properties["custom_config"]["query"]

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

    def row_to_example(formatted_row: Dict[str, Any]) -> Example:
        """Convert formatted row to TF Example without importing the main TF package."""
        feature_dict = {}

        # Add string features
        for key in ["cust_person_id", "product_id", "city", "date_of_day"]:
            value = formatted_row.get(key, "")
            feature_dict[key] = Feature(
                bytes_list=BytesList(value=[value.encode("utf-8")])
            )

        # Add float features
        for key in ["sell_val_nsp", "total_revenue"]:
            value = float(formatted_row.get(key, 0.0))
            feature_dict[key] = Feature(float_list=FloatList(value=[value]))

        return Example(features=Features(feature=feature_dict))

    return (
        pipeline
        | "ReadFromBigQuery"
        >> beam.io.ReadFromBigQuery(
            query=query,
            use_standard_sql=True,
            project=project_id,
            # Add options for large-scale processing
            use_json_exports=False,  # Use Avro for better performance with large data
            temp_dataset=f"{project_id}.temp_bq_export",  # Temporary dataset for exports
        )
        | "FormatRow" >> beam.Map(format_row)
        | "ToTFExample" >> beam.Map(row_to_example)
        | "Reshuffle" >> beam.Reshuffle()  # Add reshuffle for better parallelization
    )


class BigQueryExampleGenExecutor(BaseExampleGenExecutor):
    """Custom executor for BigQuery ExampleGen optimized for large-scale data."""

    def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
        """Returns PTransform for BigQuery to TF examples."""
        return _BigQueryToExample


def create_bigquery_example_gen(
    query: str,
    project_id: str,
    output_config: Optional[standard_artifacts.ExampleGen.OutputConfig] = None,
    beam_pipeline_args: Optional[list] = None,
) -> FileBasedExampleGen:
    """Creates a BigQuery ExampleGen component optimized for large datasets.

    Args:
        query: SQL query to execute
        project_id: GCP project ID
        output_config: Optional output configuration
        beam_pipeline_args: Optional Beam pipeline arguments for scaling

    Returns:
        FileBasedExampleGen component configured for large-scale processing
    """
    # Default beam args for large-scale processing
    default_beam_args = [
        "--runner=DataflowRunner",
        f"--project={project_id}",
        "--region=europe-west4",
        "--temp_location=gs://recs_metroua/dataflow_temp",
        "--staging_location=gs://recs_metroua/dataflow_staging",
        "--num_workers=10",  # Start with 10 workers
        "--max_num_workers=50",  # Scale up to 50 workers for large data
        "--worker_machine_type=n1-standard-4",
        "--disk_size_gb=100",
        "--use_public_ips=false",
        # Performance optimizations
        "--experiments=use_runner_v2",
        "--experiments=use_unified_worker",
        "--experiments=shuffle_mode=service",
        # Memory optimizations
        "--worker_harness_container_image=gcr.io/dataflow-templates-base/python310-template:latest",
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

    component = FileBasedExampleGen(
        input_base="dummy",  # Not used for BigQuery
        custom_config={
            "query": query,
            "project_id": project_id,
            # Add configuration for large-scale processing
            "large_scale_processing": True,
            "use_beam_sql": False,  # Use native BigQuery connector
        },
        output_config=output_config,
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            BigQueryExampleGenExecutor
        ),
    )

    # Set beam pipeline args for scaling
    if hasattr(component, "with_beam_pipeline_args"):
        component.with_beam_pipeline_args(final_beam_args)

    return component
