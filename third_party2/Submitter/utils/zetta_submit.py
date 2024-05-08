import argparse
import atexit
import shutil
import sys
from pathlib import Path

import click
import yaml
from amlt.vault import PasswordStorage
from azureml.core import Datastore, Experiment, Environment, ScriptRunConfig
from azureml.core.runconfig import RunConfiguration
from azureml.data.data_reference import DataReference
from zettasdk.util import initialize_pipeline

sys.path.append(str(Path(__file__).parent.joinpath("..")))
from utils.amlt_utils.setup_amlt import check_secrets, prepare_secrets       # noqa: E402
from utils.zetta_utils.region_manager import get_region_by_workspace         # noqa: E402
from utils.zetta_utils.region_manager import get_premium_storage_by_region   # noqa: E402
from utils.zetta_utils.region_manager import get_standard_storage_by_region  # noqa: E402


ZettAConfigFile = "zetta_config.yaml"
ZettARunnerFile = "zetta_runner.sh"


def cleanup(gen_files):
    # cleaning up generated files
    for file in gen_files:
        if Path(file).is_file():
            Path(file).unlink()


def main():
    # ------------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------------
    with open(ZettAConfigFile, "w", encoding="utf-8", newline="") as f:
        yaml.dump(args.__dict__, f, indent=2)
    atexit.register(cleanup, [ZettAConfigFile])

    help_desc = "Submit CPU job to ZettA"
    sys.argv = [
        sys.argv[0],
        "--config_path",
        ZettAConfigFile,
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
    logger.info(f"Mount Standard Azure Blob {standard_blob_name} to /datablob ...")
    storage_data = {}
    standard_datastore_name = f"ttsstandard{region}"
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
    logger.info(f"Mount Premium Azure Blob {premium_blob_name} to /modelblob ...")
    premium_datastore_name = f"ttspremium{region}"
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
        registry_name = args.docker_address.strip().split(r".")[0]
        username = check_secrets()
        if not username:
            username = click.prompt("Enter your Microsoft alias (without domain)")
            prepare_secrets(username, key_vault_name=args.key_vault_name)
        docker_username = PasswordStorage().retrieve_noninteractive(
            f"docker_{registry_name}_username", realm="meta_info", secret_type="Password")
        if not docker_username:
            raise ValueError(f"Cannot find docker username for {registry_name}, please run again.")
        docker_password = PasswordStorage().retrieve_noninteractive(
            docker_username, realm=args.docker_address, secret_type="Password")
        if not docker_password:
            raise ValueError(f"Cannot find docker password for {docker_username}, please run again.")
        zetta_env.docker.base_image_registry.username = docker_username
        zetta_env.docker.base_image_registry.password = docker_password
    logger.info(f"Docker: {args.docker_address}/{args.docker_name}")

    # Use your custom image's built-in Python environment
    zetta_env.python.user_managed_dependencies = True
    run_config.environment = zetta_env

    # Get source directory
    logger.info("source_directory: {}".format(args.local_code_dir))

    # Generate script file and ScriptRunConfig
    mount_script_name = "zetta_mount.py"
    mount_script_path = Path(__file__).absolute().parent.joinpath("zetta_utils", "zetta_mount.py")
    shutil.copy(mount_script_path, args.local_code_dir)
    atexit.register(cleanup, [mount_script_name])
    with open(ZettARunnerFile, "w", encoding="utf-8", newline="") as f:
        f.write("#!/bin/bash\n")
        f.write("set -euxo pipefail\n")
        f.write(f"python zetta_mount.py --standard-datastore-name {standard_datastore_name} ")
        f.write(f"--premium-datastore-name {premium_datastore_name}\n")
        f.write(args.cmd)
    atexit.register(cleanup, [ZettARunnerFile])
    config = ScriptRunConfig(
        source_directory=args.local_code_dir,
        run_config=run_config,
        command=["bash", ZettARunnerFile],
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
    parser.add_argument("--docker-address", type=str, default="sramdevregistry.azurecr.io",
                        help="docker registry address (default: sramdevregistry.azurecr.io)")
    parser.add_argument("--docker-name", required=True, type=str)
    parser.add_argument("--key-vault-name", type=str, default="exawatt-philly-ipgsp",
                        help="key vault name for azure docker authentication (default: exawatt-philly-ipgsp)")
    args = parser.parse_args()

    main()
