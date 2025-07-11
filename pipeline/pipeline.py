"""Metro recommendations pipeline definition."""

import os
from typing import Optional
from absl import logging
from tfx import v1 as tfx
from tfx.orchestration import pipeline
from tfx.extensions.google_cloud_big_query.example_gen.component import (
    BigQueryExampleGen,
)
from ml_metadata.proto import metadata_store_pb2

import config

# Set protobuf implementation early
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def create_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    query: str,
    project_id: str,
    region: str,
    service_account: str,
    subnetwork: str,
    enable_cache: bool = True,
    metadata_connection_config: Optional[metadata_store_pb2.ConnectionConfig] = None,
) -> pipeline.Pipeline:
    """Create TFX pipeline optimized for hybrid architecture."""

    # Use GCS paths for modules
    transform_module_path = f"{pipeline_root}/modules/transform_module.py"
    trainer_module_path = f"{pipeline_root}/modules/trainer_module.py"

    # Data ingestion - use standard BigQueryExampleGen
    example_gen = (
        BigQueryExampleGen(
            query=query,
            output_config=None,
        )
        .with_beam_pipeline_args(
            [
                f"--project={project_id}",
                f"--region={region}",
                f"--service_account_email={service_account}",
                f"--subnetwork={subnetwork}",
                "--runner=DataflowRunner",
                f"--temp_location={pipeline_root}/dataflow_temp",
                f"--staging_location={pipeline_root}/dataflow_staging",
                # Large-scale processing optimizations
                "--num_workers=5",
                "--max_num_workers=20",
                "--worker_machine_type=n1-standard-4",
                "--use_execution_time_based_autoscaling=true",
                "--no_use_public_ips",  # Fixed: use correct flag
                # Set environment variables for protobuf compatibility
                "--environment_type=DOCKER",
                "--environment_config=PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python",
            ]
        )
        .with_id("hybrid-bq-example-gen")
    )

    # Generate statistics
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    # Generate schema
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs["statistics"], infer_feature_shape=True
    )

    # Data transformation - use GCS path
    transform = tfx.components.Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=transform_module_path,
    )

    # Model training - GPU-optimized configuration
    trainer = tfx.components.Trainer(
        module_file=trainer_module_path,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=transform.outputs["post_transform_schema"],
        train_args=tfx.proto.TrainArgs(num_steps=config.TRAIN_STEPS),
        eval_args=tfx.proto.EvalArgs(num_steps=config.EVAL_STEPS),
        custom_config={
            "epochs": config.TRAIN_EPOCHS,
            "project_id": project_id,
            "products_query": config.BQ_PRODUCTS_QUERY,
            "hybrid_architecture": True,
            "training_args": [
                f"--batch-size={config.BATCH_SIZE}",
                f"--learning-rate={config.LEARNING_RATE}",
                f"--distributed-training={config.ENABLE_DISTRIBUTED_TRAINING}",
            ],
            # Vertex AI Training configuration (replaces AI Platform)
            "vertex_training_args": {
                "project": project_id,
                "region": region,
                "display_name": f"{config.PIPELINE_NAME}-training",
                "container_image_uri": config.PIPELINE_IMAGE,
                "machine_type": config.GPU_MACHINE_TYPE,
                "accelerator_type": config.GPU_ACCELERATOR_TYPE,
                "accelerator_count": config.GPU_ACCELERATOR_COUNT,
                "boot_disk_size_gb": 100,
                "enable_web_access": False,
                "service_account": service_account,
            },
        },
    )

    # Set container image for all components when running on Vertex AI
    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        transform,
        trainer,
    ]

    # Optimize Dataflow settings for data processing components
    dataflow_beam_args = [
        f"--project={project_id}",
        f"--region={region}",
        "--runner=DataflowRunner",
        f"--temp_location={pipeline_root}/dataflow_temp",
        f"--staging_location={pipeline_root}/dataflow_staging",
        f"--service_account_email={service_account}",
        f"--subnetwork={subnetwork}",
        "--no_use_public_ips",  # Fixed: use correct flag
        # Optimizations for large data processing
        "--num_workers=5",
        "--max_num_workers=20",
        "--worker_machine_type=n1-standard-2",
        "--use_execution_time_based_autoscaling=true",
        # Add protobuf environment variable for all Dataflow workers
        "--environment_type=DOCKER",
        "--environment_config=PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python",
    ]

    # Set the Beam pipeline args for components that support it
    for component in components:
        # Skip the ExampleGen as it already has its args set
        if component.id == "hybrid-bq-example-gen":
            continue

        if hasattr(component, "with_beam_pipeline_args"):
            component.with_beam_pipeline_args(dataflow_beam_args)

    pipeline_kwargs = {
        "pipeline_name": pipeline_name,
        "pipeline_root": pipeline_root,
        "components": components,
        "enable_cache": enable_cache,
    }

    if metadata_connection_config is not None:
        pipeline_kwargs["metadata_connection_config"] = metadata_connection_config

    return pipeline.Pipeline(**pipeline_kwargs)
