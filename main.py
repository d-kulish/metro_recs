"""Main entry point for Metro recommendations pipeline on Vertex AI."""

from absl import app, flags, logging
import tfx
import os
import tempfile
import shutil
import subprocess

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

        # Pre-upload modules to avoid packaging issues
        try:
            logging.info("Pre-uploading modules to GCS...")
            # Create a modules directory in GCS
            modules_path = f"{config.PIPELINE_ROOT}/modules"

            # Use gsutil to copy modules (more reliable than TF file operations)
            subprocess.run(
                [
                    "gsutil",
                    "cp",
                    "pipeline/modules/transform_module.py",
                    "pipeline/modules/trainer_module.py",
                    f"{modules_path}/",
                ],
                check=True,
            )
            logging.info(f"Modules uploaded to {modules_path}")
        except subprocess.CalledProcessError as e:
            logging.warning(f"Failed to pre-upload modules: {e}")

        # Use the correct parameter names for KubeflowV2DagRunnerConfig
        runner_config = kubeflow_v2_dag_runner.KubeflowV2DagRunnerConfig(
            display_name=config.PIPELINE_NAME,
            default_image=config.PIPELINE_IMAGE,
        )

        # Create a temporary directory for the pipeline JSON with better error handling
        temp_dir = None
        try:
            # Use /tmp instead of default temp directory for more space
            temp_dir = tempfile.mkdtemp(prefix="tfx_pipeline_", dir="/tmp")
            pipeline_json_path = os.path.join(temp_dir, f"{config.PIPELINE_NAME}.json")

            # Ensure the temp directory has proper permissions
            os.chmod(temp_dir, 0o755)

            # Set environment variables to use different temp locations
            os.environ["TMPDIR"] = temp_dir
            os.environ["TMP"] = temp_dir
            os.environ["TEMP"] = temp_dir

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

            # Clear any existing TensorFlow cached files
            try:
                tf_cache_dir = os.path.expanduser("~/.cache/tensorflow")
                if os.path.exists(tf_cache_dir):
                    shutil.rmtree(tf_cache_dir)
                    logging.info("Cleared TensorFlow cache")
            except Exception as e:
                logging.warning(f"Could not clear TF cache: {e}")

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

    # Set additional environment variables to help with file I/O
    os.environ["TF_DISABLE_MKL"] = "1"
    os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "1"

    # Clear any existing temporary files
    import glob

    temp_patterns = ["/tmp/tfx_*", "/tmp/tmp*tfx*", os.path.expanduser("~/tmp*tfx*")]

    for pattern in temp_patterns:
        for temp_file in glob.glob(pattern):
            try:
                if os.path.isdir(temp_file):
                    shutil.rmtree(temp_file)
                else:
                    os.remove(temp_file)
                logging.info(f"Cleaned up existing temp file: {temp_file}")
            except Exception as e:
                logging.debug(f"Could not clean up {temp_file}: {e}")

    # Ensure we have enough disk space by checking available space
    import shutil as disk_util

    total, used, free = disk_util.disk_usage("/")
    free_gb = free // (1024**3)
    logging.info(f"Available disk space: {free_gb} GB")

    if free_gb < 10:  # Less than 10GB free
        logging.error(
            f"Insufficient disk space: {free_gb} GB available. Need at least 10GB."
        )
        raise RuntimeError(f"Insufficient disk space: {free_gb} GB")

    run_pipeline()


if __name__ == "__main__":
    app.run(main)
