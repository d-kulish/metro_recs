"""Main entry point for Metro recommendations pipeline on Vertex AI."""

from absl import app, flags, logging
import tfx

from pipeline.pipeline import create_pipeline
import config

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "runner", "local", ["local", "vertex"], "Pipeline runner to use."
)


def run_pipeline():
    """Run the TFX pipeline."""

    runner = None
    metadata_config = None

    if FLAGS.runner == "vertex":
        from tfx.orchestration.experimental.kubeflow.v2.kubeflow_v2_dag_runner import (
            KubeflowV2DagRunner,
            KubeflowV2DagRunnerConfig,
        )

        runner_config = KubeflowV2DagRunnerConfig(display_name=config.PIPELINE_NAME)
        runner = KubeflowV2DagRunner(config=runner_config)
        metadata_config = tfx.orchestration.experimental.get_default_vertex_metadata_config()
    else:
        from tfx.orchestration.local.local_dag_runner import LocalDagRunner

        runner = LocalDagRunner()

    pipeline = create_pipeline(
        pipeline_name=config.PIPELINE_NAME,
        pipeline_root=config.PIPELINE_ROOT,
        query=config.BQ_QUERY,
        project_id=config.PROJECT_ID,
        region=config.LOCATION,
        metadata_connection_config=metadata_config,
    )

    logging.info(f"Running pipeline '{config.PIPELINE_NAME}' with runner '{FLAGS.runner}'")
    logging.info(f"Pipeline root: {config.PIPELINE_ROOT}")

    runner.run(pipeline)


def main(_):
    logging.set_verbosity(logging.INFO)
    run_pipeline()


if __name__ == "__main__":
    app.run(main)
