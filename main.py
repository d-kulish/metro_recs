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
        # For TFX >= 1.14, KubeflowV2DagRunner is the recommended runner for Vertex AI.
        # We try it first.
        try:
            from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner

            # The config now specifies the default image for all components.
            runner_config = kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
                display_name=config.PIPELINE_NAME,
                default_image=config.PIPELINE_IMAGE,
            )
            runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
                config=runner_config,
                output_dir=config.PIPELINE_ROOT,
            )
            logging.info("Using KubeflowV2DagRunner for Vertex AI.")

        except (ImportError, TypeError) as e:
            logging.warning(f"Could not use KubeflowV2DagRunner, falling back to legacy runner. Error: {e}")
            try:
                # Fallback to the legacy VertexRunner
                from tfx.extensions.google_cloud_ai_platform.runner import vertex_runner

                runner = vertex_runner.VertexRunner(
                    project_id=config.VERTEX_PROJECT_ID,
                    region=config.VERTEX_REGION,
                    image_uri=config.PIPELINE_IMAGE,
                )
                logging.info("Using legacy VertexRunner for Vertex AI.")

            except ImportError:
                logging.fatal(
                    "Neither KubeflowV2DagRunner nor the legacy VertexRunner are available. "
                    "Please check your TFX installation and dependencies."
                )
                return

        # Try to get metadata config
        try:
            from tfx.orchestration.experimental import (
                get_default_vertex_metadata_config,
            )

            metadata_config = get_default_vertex_metadata_config()
        except (ImportError, AttributeError):
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
