"""Main entry point for Metro recommendations pipeline on Vertex AI."""

from absl import app, flags, logging
import tfx

from pipeline.pipeline import create_pipeline
import config

FLAGS = flags.FLAGS
flags.DEFINE_enum("runner", "vertex", ["local", "vertex"], "Pipeline runner to use.")


def run_pipeline():
    """Run the TFX pipeline."""

    runner = None
    metadata_config = None

    if FLAGS.runner == "vertex":
        try:
            # Try the newer import path first
            from tfx.orchestration.vertex.vertex_dag_runner import VertexDagRunner
            from tfx.orchestration.vertex.vertex_dag_runner import VertexDagRunnerConfig

            runner_config = VertexDagRunnerConfig(
                display_name=config.PIPELINE_NAME,
                project_id=config.VERTEX_PROJECT_ID,
                default_image_uri=config.PIPELINE_IMAGE,
            )
            runner = VertexDagRunner(config=runner_config)

        except ImportError:
            # Fallback to the experimental import if the above fails
            try:
                from tfx.orchestration.experimental.kubeflow.v2.kubeflow_v2_dag_runner import (
                    KubeflowV2DagRunner,
                    KubeflowV2DagRunnerConfig,
                )

                runner_config = KubeflowV2DagRunnerConfig(
                    display_name=config.PIPELINE_NAME,
                    project_id=config.VERTEX_PROJECT_ID,
                    default_image_uri=config.PIPELINE_IMAGE,
                )
                runner = KubeflowV2DagRunner(config=runner_config)

            except ImportError:
                raise ImportError(
                    "Unable to import Vertex AI runner. Please check your TFX installation."
                )

        # Get metadata config
        try:
            metadata_config = (
                tfx.orchestration.experimental.get_default_vertex_metadata_config()
            )
        except AttributeError:
            # If the experimental method doesn't exist, try the newer location
            try:
                from tfx.orchestration.vertex import vertex_metadata_config

                metadata_config = (
                    vertex_metadata_config.get_default_vertex_metadata_config()
                )
            except ImportError:
                logging.warning("Could not configure Vertex metadata. Using default.")
                metadata_config = None
    else:
        from tfx.orchestration.local.local_dag_runner import LocalDagRunner

        runner = LocalDagRunner()
        # Don't set metadata_config for local runs - let it use default

    pipeline = create_pipeline(
        pipeline_name=config.PIPELINE_NAME,
        pipeline_root=config.PIPELINE_ROOT,
        query=config.BQ_QUERY,
        project_id=config.PROJECT_ID,
        region=config.LOCATION,
        metadata_connection_config=metadata_config,
    )

    logging.info(
        f"Running pipeline '{config.PIPELINE_NAME}' with runner '{FLAGS.runner}'"
    )
    logging.info(f"Pipeline root: {config.PIPELINE_ROOT}")

    runner.run(pipeline)


def main(_):
    logging.set_verbosity(logging.INFO)
    run_pipeline()


if __name__ == "__main__":
    app.run(main)
