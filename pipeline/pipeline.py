"""Metro recommendations pipeline definition."""

import os
from typing import Optional
from absl import logging
from tfx import v1 as tfx
from tfx.orchestration import pipeline
from ml_metadata.proto import metadata_store_pb2

import config


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
    """Create TFX pipeline for Metro recommendations.

    Args:
        pipeline_name: Name of the pipeline
        pipeline_root: Root path for pipeline artifacts
        query: BigQuery SQL query
        project_id: GCP project ID
        region: GCP region
        service_account: Service account email for Dataflow workers
        subnetwork: Dataflow subnetwork
        enable_cache: Whether to enable caching
        metadata_connection_config: Metadata connection config

    Returns:
        TFX Pipeline instance
    """

    # Data ingestion for training examples (user-product interactions)
    example_gen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(
        query=query
    ).with_id("bq-interactions-gen")

    # Generate statistics
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    # Generate schema
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs["statistics"], infer_feature_shape=True
    )

    # Data transformation
    transform = tfx.components.Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.abspath("pipeline/modules/transform_module.py"),
    )

    # Model training with GPU support using Vertex AI Training
    # Use the standard Trainer component with corrected AI Platform configuration
    trainer = tfx.components.Trainer(
        module_file=os.path.abspath("pipeline/modules/trainer_module.py"),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=transform.outputs["post_transform_schema"],
        train_args=tfx.proto.TrainArgs(num_steps=config.TRAIN_STEPS),
        eval_args=tfx.proto.EvalArgs(num_steps=config.EVAL_STEPS),
        custom_config={
            "epochs": config.TRAIN_EPOCHS,
            "project_id": project_id,
            "products_query": config.BQ_PRODUCTS_QUERY,
            # Use corrected AI Platform Training configuration
            "ai_platform_training_args": {
                "project": project_id,
                "region": region,
                "jobDir": f"{pipeline_root}/training_jobs",
                "args": [
                    f"--distributed-training={config.ENABLE_DISTRIBUTED_TRAINING}",
                    f"--batch-size={config.BATCH_SIZE}",
                    f"--learning-rate={config.LEARNING_RATE}",
                ],
                "masterConfig": {
                    "imageUri": config.PIPELINE_IMAGE,
                    "machineType": config.GPU_MACHINE_TYPE,  # Changed from machine_type
                    "acceleratorConfig": {
                        "type": config.GPU_ACCELERATOR_TYPE,
                        "count": config.GPU_ACCELERATOR_COUNT,
                    },
                },
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

    # Set the Beam pipeline args for components that support it.
    # This is needed to run the Beam-based components on Dataflow.
    for component in components:
        # Not all components are Beam-based and support this method.
        if hasattr(component, "with_beam_pipeline_args"):
            component.with_beam_pipeline_args(
                [
                    f"--project={project_id}",
                    f"--region={region}",
                    "--runner=DataflowRunner",
                    f"--temp_location={pipeline_root}/temp",
                    f"--staging_location={pipeline_root}/staging",
                    f"--service_account_email={service_account}",
                    f"--subnetwork={subnetwork}",
                    # Prevent Dataflow workers from using public IPs. This is a common requirement
                    # in secure VPC environments and relies on Private Google Access being enabled.
                    "--no_use_public_ips",
                ]
            )

    pipeline_kwargs = {
        "pipeline_name": pipeline_name,
        "pipeline_root": pipeline_root,
        "components": components,
        "enable_cache": enable_cache,
    }

    # Only add metadata_connection_config if it's not None
    if metadata_connection_config is not None:
        pipeline_kwargs["metadata_connection_config"] = metadata_connection_config

    return pipeline.Pipeline(**pipeline_kwargs)
