"""Main entry point for Metro recommendations pipeline on Vertex AI."""

from absl import app, flags, logging
import tfx
import os
import tempfile
import shutil

from pipeline.pipeline import create_pipeline
import config

FLAGS = flags.FLAGS
flags.DEFINE_enum("runner", "vertex", ["local", "vertex"], "Pipeline runner to use.")


def run_pipeline():
    """Run the TFX pipeline."""

    runner = None
    metadata_config = None

    if FLAGS.runner == "vertex":
        # For Vertex AI, we need to compile and submit the pipeline
        from tfx.orchestration.kubeflow.v2 import kubeflow_v2_dag_runner
        from google.cloud import aiplatform

        # Use the correct parameter names for KubeflowV2DagRunnerConfig
        runner_config = kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
            display_name=config.PIPELINE_NAME,
            default_image=config.PIPELINE_IMAGE,
        )

        # Create a temporary directory for the pipeline JSON with better error handling
        temp_dir = None
        try:
            # Create temp directory with more permissive settings
            temp_dir = tempfile.mkdtemp(prefix="tfx_pipeline_")
            pipeline_json_path = os.path.join(temp_dir, f"{config.PIPELINE_NAME}.json")

            # Ensure the temp directory has proper permissions
            os.chmod(temp_dir, 0o755)

            runner = kubeflow_v2_dag_runner.KubeflowV2DagRunner(
                config=runner_config,
                output_filename=pipeline_json_path,
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

            # Create the pipeline
            pipeline = create_pipeline(
                pipeline_name=config.PIPELINE_NAME,
                pipeline_root=config.PIPELINE_ROOT,
                query=config.BQ_QUERY,
                project_id=config.PROJECT_ID,
                region=config.VERTEX_REGION,
                service_account=config.VERTEX_SERVICE_ACCOUNT,
                subnetwork=config.DATAFLOW_SUBNETWORK,
                metadata_connection_config=metadata_config,
            )

            logging.info(
                f"Compiling pipeline '{config.PIPELINE_NAME}' with runner '{FLAGS.runner}'"
            )
            logging.info(f"Pipeline root: {config.PIPELINE_ROOT}")
            logging.info(f"Temp directory: {temp_dir}")

            # Compile the pipeline with additional error handling
            try:
                runner.run(pipeline)
            except Exception as e:
                logging.error(f"Pipeline compilation failed: {e}")
                # Try to clean up any partial files
                if os.path.exists(pipeline_json_path):
                    os.remove(pipeline_json_path)
                raise

            # Now submit the compiled pipeline to Vertex AI
            logging.info("Submitting pipeline to Vertex AI...")

            # Initialize Vertex AI client
            aiplatform.init(
                project=config.VERTEX_PROJECT_ID,
                location=config.VERTEX_REGION,
            )

            # Submit the pipeline
            job = aiplatform.PipelineJob(
                display_name=config.PIPELINE_NAME,
                template_path=pipeline_json_path,
                pipeline_root=config.PIPELINE_ROOT,
                project=config.VERTEX_PROJECT_ID,
                location=config.VERTEX_REGION,
                enable_caching=True,
                parameter_values={},
                job_id=None,
            )

            try:
                logging.info(f"Submitting pipeline job: {config.PIPELINE_NAME}")
                # Pass the service account to ensure the pipeline runs with the correct permissions.
                job.submit(service_account=config.VERTEX_SERVICE_ACCOUNT)
                logging.info(
                    f"Pipeline submitted successfully. Job name: {job.resource_name}"
                )
                # The dashboard URI is available after the job object is created/submitted.
                logging.info(f"You can view the pipeline at: {job._dashboard_uri()}")
            except Exception as e:
                # Provide more detailed error logging.
                logging.error(f"Failed to submit pipeline: {e}", exc_info=True)
                raise e

        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logging.info(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    logging.warning(
                        f"Failed to clean up temp directory {temp_dir}: {e}"
                    )

    else:
        from tfx.orchestration.local.local_dag_runner import LocalDagRunner

        runner = LocalDagRunner()

        pipeline = create_pipeline(
            pipeline_name=config.PIPELINE_NAME,
            pipeline_root=config.PIPELINE_ROOT,
            query=config.BQ_QUERY,
            project_id=config.PROJECT_ID,
            region=config.VERTEX_REGION,
            service_account=config.VERTEX_SERVICE_ACCOUNT,
            subnetwork=config.DATAFLOW_SUBNETWORK,
            metadata_connection_config=None,
        )

        logging.info(
            f"Running pipeline '{config.PIPELINE_NAME}' with runner '{FLAGS.runner}'"
        )
        logging.info(f"Pipeline root: {config.PIPELINE_ROOT}")

        runner.run(pipeline)


def main(_):
    logging.set_verbosity(logging.INFO)

    # Set TensorFlow environment variables to avoid file I/O issues
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

    # Ensure we have enough disk space by checking available space
    import shutil as disk_util

    total, used, free = disk_util.disk_usage("/")
    free_gb = free // (1024**3)
    logging.info(f"Available disk space: {free_gb} GB")

    if free_gb < 5:  # Less than 5GB free
        logging.warning(
            f"Low disk space: {free_gb} GB available. This may cause file I/O issues."
        )

    run_pipeline()


if __name__ == "__main__":
    app.run(main)
