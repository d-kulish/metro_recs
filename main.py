"""Main entry point for Metro recommendations pipeline on Vertex AI."""

import os
from absl import app, flags, logging
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from tfx.orchestration.experimental.interactive.interactive_context import (
    InteractiveContext,
)
from pipeline.pipeline import create_pipeline
import config

FLAGS = flags.FLAGS


def run_pipeline():
    """Run the pipeline locally first, then can be adapted for Vertex AI."""

    # Use local runner for development and testing
    runner = LocalDagRunner()

    pipeline = create_pipeline(
        pipeline_name=config.PIPELINE_NAME,
        pipeline_root=config.PIPELINE_ROOT,
        query=config.BQ_QUERY,
        project_id=config.PROJECT_ID,
        region=config.VERTEX_REGION,
    )

    logging.info(f"Running pipeline: {config.PIPELINE_NAME}")
    logging.info(f"Pipeline root: {config.PIPELINE_ROOT}")

    runner.run(pipeline)


def main(_):
    logging.set_verbosity(logging.INFO)
    run_pipeline()


if __name__ == "__main__":
    app.run(main)
    logging.set_verbosity(logging.INFO)
    run_pipeline()


if __name__ == "__main__":
    app.run(main)
