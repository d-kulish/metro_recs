"""Main entry point for Metro recommendations pipeline on Vertex AI."""

import os
from absl import app, flags, logging
from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
from pipeline.pipeline import create_pipeline
import config

FLAGS = flags.FLAGS


def run_pipeline():
    """Run the pipeline on Vertex AI."""

    runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
        config=kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
            project_id=config.VERTEX_PROJECT_ID,
            display_name=config.PIPELINE_NAME,
            default_image=f"gcr.io/{config.VERTEX_PROJECT_ID}/tfx-pipeline",
        ),
        output_filename=f"{config.PIPELINE_NAME}.json",
    )

    pipeline = create_pipeline(
        pipeline_name=config.PIPELINE_NAME,
        pipeline_root=config.PIPELINE_ROOT,
        query=config.BQ_QUERY,
        project_id=config.PROJECT_ID,
        region=config.VERTEX_REGION,
    )

    runner.run(
        pipeline=pipeline,
        project_id=config.VERTEX_PROJECT_ID,
        region=config.VERTEX_REGION,
    )


def main(_):
    logging.set_verbosity(logging.INFO)
    run_pipeline()


if __name__ == "__main__":
    app.run(main)
