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
        enable_cache: Whether to enable caching
        metadata_connection_config: Metadata connection config

    Returns:
        TFX Pipeline instance
    """

    # Data ingestion from BigQuery
    example_gen = tfx.extensions.google_cloud_big_query.BigQueryExampleGen(query=query)

    # Generate statistics
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs["examples"]
    )

    # Generate schema
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs["statistics"], infer_feature_shape=False
    )

    # Data transformation
    transform = tfx.components.Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.abspath("pipeline/modules/transform_module.py"),
    )

    # Model training
    trainer = tfx.components.Trainer(
        module_file=os.path.abspath("pipeline/modules/trainer_module.py"),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=tfx.proto.TrainArgs(num_steps=config.TRAIN_STEPS),
        eval_args=tfx.proto.EvalArgs(num_steps=config.EVAL_STEPS),
        custom_config={
            "epochs": config.TRAIN_EPOCHS,
        },
    )

    # Model evaluation
    evaluator = tfx.components.Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
    )

    # Model pusher (deployment)
    pusher = tfx.components.Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=os.path.join(pipeline_root, "serving_model")
            )
        ),
    )

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        transform,
        trainer,
        evaluator,
        pusher,
    ]

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=enable_cache,
        metadata_connection_config=metadata_connection_config,
    )
