"""Main entry point for Metro recommendations pipeline on Vertex AI."""

from absl import app, flags, logging
import tfx
import os
import tempfile

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

        # Create a temporary directory for the pipeline JSON
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline_json_path = os.path.join(temp_dir, f"{config.PIPELINE_NAME}.json")

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

            # Compile the pipeline
            runner.run(pipeline)

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
                # Configure GPU resources for the entire pipeline
                parameter_values={},
                # Set the machine type and accelerator for components that support it
                job_id=None,
            )

            # Configure GPU resources at submission time
            job._gca_resource.spec.task_details[0].executor_label = "trainer"
            job._gca_resource.spec.task_details[
                0
            ].task_spec.container_spec.resources.accelerator.type = "NVIDIA_TESLA_T4"
            job._gca_resource.spec.task_details[
                0
            ].task_spec.container_spec.resources.accelerator.count = 1

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
    run_pipeline()


if __name__ == "__main__":
    app.run(main)
