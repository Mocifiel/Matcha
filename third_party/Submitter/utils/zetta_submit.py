import argparse
import atexit
import json
import os
import random
import shutil
import string
import sys
import uuid
from pathlib import Path

import yaml
from amlt.globals import CONFIG_HOME
from azureml.core import Datastore, Experiment, Environment, ScriptRunConfig
from azureml.core.runconfig import RunConfiguration
from azureml.data.data_reference import DataReference
from zettasdk.util import initialize_pipeline

sys.path.append(str(Path(__file__).parent.joinpath("..")))
from utils.zetta_utils.region_manager import get_region_by_workspace  # noqa: E402
from utils.zetta_utils.region_manager import get_premium_storage_by_region  # noqa: E402
from utils.zetta_utils.region_manager import get_standard_storage_by_region  # noqa: E402


def cleanup(gen_files):
    # cleaning up generated files
    for file in gen_files:
        if Path(file).is_file():
            Path(file).unlink()


def get_docker_password(config_file, docker_registry, docker_username):
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    key = f"{docker_registry}/{docker_username}"
    if key not in config:
        raise ValueError(
            f">>> Cannot get the docker password for {key}, please run a GPU job to set up the env firstly."
        )
    key_vault_id = config[key]
    get_secret_cmd = f"az keyvault secret show --id {key_vault_id}"
    # tempfile.NamedTemporaryFil doesn't work with the redirection operator on Windows
    temp_file = str(Path(__file__).parent.joinpath(f"temp_{uuid.uuid4()}.json").resolve())
    atexit.register(cleanup, [temp_file])
    result = os.system(get_secret_cmd + f" > {temp_file}")
    if result != 0:
        azure_auth_cmd = "az login --use-device-code"
        os.system(azure_auth_cmd)
        os.system(get_secret_cmd + f" > {temp_file}")
    with open(temp_file, "r", encoding="utf-8") as f_json:
        secret = json.load(f_json)
    return secret["value"]


def main():
    random_string = "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
    # ------------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------------
    zetta_config_path = str(Path(args.local_code_dir).absolute().joinpath(f"zetta_config_{random_string}.yaml"))
    with open(zetta_config_path, "w", encoding="utf-8", newline="") as f:
        yaml.dump(args.__dict__, f, indent=2)
    atexit.register(cleanup, [zetta_config_path])

    help_desc = "Submit CPU job to ZettA"
    sys.argv = [
        sys.argv[0],
        "--config_path",
        zetta_config_path,
        "--experiment_name",
        args.experiment_name,
    ]
    workspace, _, _, _, _, _, logger = initialize_pipeline(help_desc, default_workspace_name=args.workspace_name)

    # ------------------------------------------------------------------------
    # RunConfiguration
    # ------------------------------------------------------------------------
    logger.info("Setting up ScriptRunConfig definitions ...")
    run_config = RunConfiguration()
    run_config.target = args.compute_target
    region = get_region_by_workspace(args.workspace_name)
    logger.info(f"Data storage region: {region}")

    # ------------------------------------------------------------------------
    # Mount Standard Azure Blob
    # ------------------------------------------------------------------------
    standard_blob_name = get_standard_storage_by_region(region)
    standard_blob_container = args.data_blob_container
    logger.info(f'Mount "{standard_blob_container}" in "{standard_blob_name}" to /datablob ...')
    storage_data = {}
    standard_datastore_name = (
        f"submitter_{standard_blob_name.replace('-', '_')}_{standard_blob_container.replace('-', '_')}"
    )
    standard_datastore = Datastore.get(workspace, standard_datastore_name)
    standard_data_reference = DataReference(
        datastore=standard_datastore,
        data_reference_name=standard_datastore_name,
        path_on_datastore="/",
        mode="mount",
    )
    storage_data[standard_datastore_name] = standard_data_reference.to_config()

    # ------------------------------------------------------------------------
    # Mount Premium Azure Blob
    # ------------------------------------------------------------------------
    premium_blob_name = get_premium_storage_by_region(region)
    premium_blob_container = args.model_blob_container
    logger.info(f'Mount "{premium_blob_container}" in "{premium_blob_name}" to /modelblob ...')
    premium_datastore_name = (
        f"submitter_{premium_blob_name.replace('-', '_')}_{premium_blob_container.replace('-', '_')}"
    )
    premium_datastore = Datastore.get(workspace, premium_datastore_name)
    premium_data_reference = DataReference(
        datastore=premium_datastore,
        data_reference_name=premium_datastore_name,
        path_on_datastore="/",
        mode="mount",
    )
    storage_data[premium_datastore_name] = premium_data_reference.to_config()
    run_config.data_references = storage_data

    # ------------------------------------------------------------------------
    # Docker image configuration
    # ------------------------------------------------------------------------
    logger.info("Setting up Docker image configuration ...")
    zetta_env = Environment(name="TtsZettAEnv")
    zetta_env.docker.base_image = args.docker_name
    # Set the container registry information.
    if r"docker.io" not in args.docker_address:
        zetta_env.docker.base_image_registry.address = args.docker_address
        config_file = Path(CONFIG_HOME).joinpath("vault.yml")
        if not config_file.is_file():
            print(">>> Please run a GPU job using AMLT to set up the ENV firstly.")
            exit(0)

        docker_password = get_docker_password(config_file, args.docker_address, args.docker_username)
        zetta_env.docker.base_image_registry.username = args.docker_username
        zetta_env.docker.base_image_registry.password = docker_password
    logger.info(f"Docker: {args.docker_address}/{args.docker_name}")

    # Use your custom image's built-in Python environment
    zetta_env.python.user_managed_dependencies = True
    run_config.environment = zetta_env

    # Get source directory
    logger.info("source_directory: {}".format(args.local_code_dir))

    # Generate script file and ScriptRunConfig
    mount_script_name = "zetta_mount"
    mount_script_path = Path(__file__).absolute().parent.joinpath("zetta_utils", f"{mount_script_name}.py")
    mount_script_dest_name = f"{mount_script_name}_{random_string}.py"
    mount_script_dest_path = Path(args.local_code_dir).absolute().joinpath(mount_script_dest_name)
    shutil.copy(mount_script_path, mount_script_dest_path)
    atexit.register(cleanup, [mount_script_name])
    zetta_runner_file_name = f"zetta_runner_{random_string}.sh"
    zetta_runner_file_path = str(Path(args.local_code_dir).absolute().joinpath(zetta_runner_file_name))
    with open(zetta_runner_file_path, "w", encoding="utf-8", newline="") as f:
        f.write("#!/bin/bash\n")
        f.write("set -euxo pipefail\n")
        f.write(f"python {mount_script_dest_name} --standard-datastore-name {standard_datastore_name} ")
        f.write(f"--premium-datastore-name {premium_datastore_name}\n")
        f.write(args.cmd)
    atexit.register(cleanup, [zetta_runner_file_path, mount_script_dest_path])
    config = ScriptRunConfig(
        source_directory=args.local_code_dir,
        run_config=run_config,
        command=["bash", zetta_runner_file_name],
    )

    logger.info("Submitting job ...")
    experiment = Experiment(workspace, args.experiment_name)
    run = experiment.submit(config)
    run.display_name = args.display_name
    logger.info(f"Experiment name: {args.experiment_name}")
    logger.info(f"Display name: {args.display_name}")
    logger.info(f"AML Portal URL: {run.get_portal_url()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-name", required=True, type=str)
    parser.add_argument("--compute-target", required=True, type=str)
    parser.add_argument("--experiment-name", required=True, type=str)
    parser.add_argument("--display-name", required=True, type=str)
    parser.add_argument("--cmd", required=True, type=str)
    parser.add_argument("--local-code-dir", required=True, type=str)
    parser.add_argument("--data-blob-container", default="philly-ipgsp", type=str)
    parser.add_argument("--model-blob-container", default="philly-ipgsp", type=str)
    parser.add_argument(
        "--docker-address",
        type=str,
        default="sramdevregistry.azurecr.io",
        help="docker registry address (default: sramdevregistry.azurecr.io)",
    )
    parser.add_argument("--docker-name", required=True, type=str)
    parser.add_argument(
        "--key-vault-name",
        type=str,
        default="exawatt-philly-ipgsp",
        help="key vault name for azure docker authentication (default: exawatt-philly-ipgsp)",
    )
    parser.add_argument(
        "--docker-username",
        type=str,
        default="tts-itp-user",
        help="docker username for azure docker authentication (default: tts-itp-user)",
    )
    args = parser.parse_args()

    main()
