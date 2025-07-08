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
            # For TFX 1.15.0, try the Google Cloud AI Platform runner
            from tfx.extensions.google_cloud_ai_platform.runner import vertex_runner

            runner = vertex_runner.VertexRunner(
                project_id=config.VERTEX_PROJECT_ID,
                region=config.VERTEX_REGION,
                image_uri=config.PIPELINE_IMAGE,
            )

        except ImportError:
            try:
                # Alternative: Try the Kubeflow V2 runner which should be available
                from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner

                # Use the correct parameter names for KubeflowV2DagRunnerConfig
                runner_config = kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
                    display_name=config.PIPELINE_NAME,
                    default_image_uri=config.PIPELINE_IMAGE,
                )
                runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
                    config=runner_config,
                    output_dir=config.PIPELINE_ROOT,
                )

            except ImportError:
                # Final fallback: Try the beam runner with DataflowRunner
                from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
                from apache_beam.options.pipeline_options import PipelineOptions

                beam_options = PipelineOptions(
                    [
                        f"--project={config.VERTEX_PROJECT_ID}",
                        f"--region={config.VERTEX_REGION}",
                        "--runner=DataflowRunner",
                        f"--temp_location={config.PIPELINE_ROOT}/temp",
                        f"--staging_location={config.PIPELINE_ROOT}/staging",
                    ]
                )

                runner = BeamDagRunner(
                    beam_pipeline_args=beam_options.get_all_options()
                )

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
