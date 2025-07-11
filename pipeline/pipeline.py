"""Metro recommendations pipeline definition."""

import os
from typing import Optional
from absl import logging
from tfx import v1 as tfx
from tfx.orchestration import pipeline
from pipeline.components.bq_example_gen import create_bigquery_example_gen
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
    """Create TFX pipeline for Metro recommendations."""

    # Use GCS paths for modules to avoid local file packaging issues
    transform_module_path = f"{pipeline_root}/modules/transform_module.py"
    trainer_module_path = f"{pipeline_root}/modules/trainer_module.py"

    # Data ingestion using the custom BigQueryExampleGen for large-scale processing
    example_gen = create_bigquery_example_gen(
        query=query,
        project_id=project_id,
    ).with_id("custom-bq-interactions-gen")

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

    # Model training - use GCS path
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
            # Training arguments
            "training_args": [
                f"--distributed-training={config.ENABLE_DISTRIBUTED_TRAINING}",
                f"--batch-size={config.BATCH_SIZE}",
                f"--learning-rate={config.LEARNING_RATE}",
            ],
            # Force GPU configuration
            "ai_platform_training_args": {
                "project": project_id,
                "region": region,
                "runtimeVersion": "2.15",
                "pythonVersion": "3.10",
                "jobDir": f"{pipeline_root}/training_jobs",
                "masterConfig": {
                    "imageUri": config.PIPELINE_IMAGE,
                    "machineType": config.GPU_MACHINE_TYPE,
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
    for component in components:
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
                    "--no_use_public_ips",
                ]
            )

    pipeline_kwargs = {
        "pipeline_name": pipeline_name,
        "pipeline_root": pipeline_root,
        "components": components,
        "enable_cache": enable_cache,
    }

    if metadata_connection_config is not None:
        pipeline_kwargs["metadata_connection_config"] = metadata_connection_config

    return pipeline.Pipeline(**pipeline_kwargs)
