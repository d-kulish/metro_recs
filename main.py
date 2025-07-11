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


def cleanup_disk_space():
    """Aggressively clean up disk space."""
    logging.info("Starting aggressive disk cleanup...")

    cleanup_commands = [
        # Clean apt cache
        "sudo apt-get clean",
        # Clean pip cache
        "pip cache purge",
        # Clean conda cache if available
        "conda clean -a -y || true",
        # Remove old log files
        "sudo find /var/log -name '*.log' -type f -mtime +7 -delete || true",
        # Clean temp directories
        "sudo rm -rf /tmp/* || true",
        "sudo rm -rf /var/tmp/* || true",
        # Clean Docker if available
        "docker system prune -f || true",
    ]

    for cmd in cleanup_commands:
        try:
            subprocess.run(cmd, shell=True, check=False)
            logging.info(f"Executed: {cmd}")
        except Exception as e:
            logging.warning(f"Failed to execute {cmd}: {e}")

    # Clean Python cache files
    import glob

    cache_patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        os.path.expanduser("~/.cache/*"),
        "/tmp/tmp*",
    ]

    for pattern in cache_patterns:
        for cache_file in glob.glob(pattern, recursive=True):
            try:
                if os.path.isdir(cache_file):
                    shutil.rmtree(cache_file)
                else:
                    os.remove(cache_file)
                logging.debug(f"Removed cache: {cache_file}")
            except Exception as e:
                logging.debug(f"Could not remove {cache_file}: {e}")


def check_and_free_disk_space():
    """Check disk space and attempt to free up space if needed."""
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (1024**3)
    used_gb = used // (1024**3)
    total_gb = total // (1024**3)

    logging.info(
        f"Disk usage: {used_gb} GB used / {total_gb} GB total ({free_gb} GB free)"
    )

    if free_gb < 5:
        logging.warning(f"Low disk space detected: {free_gb} GB free")
        cleanup_disk_space()

        # Check again after cleanup
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (1024**3)
        logging.info(f"After cleanup: {free_gb} GB free")

    return free_gb


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

        # Create a minimal temporary directory
        temp_dir = None
        try:
            # Use minimal temp space - just for the JSON file
            temp_dir = tempfile.mkdtemp(prefix="tfx_", dir="/tmp")
            pipeline_json_path = os.path.join(temp_dir, f"{config.PIPELINE_NAME}.json")

            os.chmod(temp_dir, 0o755)

            # Set minimal environment variables
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

            logging.info(f"Compiling pipeline '{config.PIPELINE_NAME}'")
            logging.info(f"Pipeline root: {config.PIPELINE_ROOT}")

            # Compile the pipeline
            try:
                runner.run(pipeline)
            except Exception as e:
                logging.error(f"Pipeline compilation failed: {e}")
                if os.path.exists(pipeline_json_path):
                    os.remove(pipeline_json_path)
                raise

            # Submit the pipeline to Vertex AI
            logging.info("Submitting pipeline to Vertex AI...")

            aiplatform.init(
                project=config.VERTEX_PROJECT_ID,
                location=config.VERTEX_REGION,
            )

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
                job.submit(service_account=config.VERTEX_SERVICE_ACCOUNT)
                logging.info(
                    f"Pipeline submitted successfully. Job name: {job.resource_name}"
                )
                logging.info(f"You can view the pipeline at: {job._dashboard_uri()}")
            except Exception as e:
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

    # Check and free up disk space first
    free_gb = check_and_free_disk_space()

    if free_gb < 2:  # Reduced minimum requirement
        logging.error(f"Critical: Only {free_gb} GB available. Cannot proceed safely.")
        logging.error("Please free up disk space manually:")
        logging.error("1. Run: sudo apt-get clean")
        logging.error("2. Run: pip cache purge")
        logging.error("3. Remove large files from ~/")
        logging.error("4. Run: docker system prune -f (if Docker is installed)")
        raise RuntimeError(f"Insufficient disk space: {free_gb} GB")

    # Set TensorFlow environment variables
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    os.environ["TF_DISABLE_MKL"] = "1"
    os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = "1"

    run_pipeline()


if __name__ == "__main__":
    app.run(main)
