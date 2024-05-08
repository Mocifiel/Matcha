#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2020  Microsoft (author: Ke Wang)

import argparse
import atexit
import copy
import platform
import random
import string
import sys
import subprocess
import yaml
from pathlib import Path

sys.path.append(str(Path(__file__).parent.joinpath("..")))
from utils.amlt_utils.skumanager import get_model_storage_by_region_name  # noqa: E402
from utils.amlt_utils.skumanager import get_data_storage_by_region_name  # noqa: E402

# from utils.amlt_utils.skumanager import get_persistent_storage_by_region_name  # noqa: E402
from utils.amlt_utils.setup_amlt import check_setup_amulet, get_config_home  # noqa: E402
from utils.misc.common import str_to_bool  # noqa: E402

SHELL = False if platform.system() == "Windows" else True


def cleanup(gen_files):
    # cleaning up generated files
    for file in gen_files:
        if Path(file).is_file():
            Path(file).unlink()


def parse_extra_params(extra_params, exp_name):
    r"""parse extra parameters
    Args:
        extra_params: parameters for jobs and separated with & and ;
            (jobname1&--data data1;jobname2&--data data2)
    Return:
        output: job name and parameters for each job

    Examples:
        >>> extra_params = "jobname1&--data data1;jobname2&--data data2"
        >>> jobs_params = parse_extra_params(extra_params)
    """
    jobs_params = []

    # when extra parameter is empty
    if len(extra_params) == 0:
        name = f"{exp_name}"
        params = ""
        jobs_params.append((name, params))
    else:
        # using ";" to separate parameters for each job
        extra_params = extra_params.strip().split(";;;")
        for idx in range(len(extra_params)):
            # using "&" to separate job name and parameters
            item = extra_params[idx].strip().split("&&&")
            if len(item) == 2:
                name, params = item
            elif len(item) == 1:
                # to handle the last job
                if len(item[0]) == 0:
                    continue
                # assume job name is not provided
                name, params = f"{exp_name}", item
            else:
                raise ValueError("The format of extra_params is wrong." f"extra_params: {extra_params}")

            params = params_list_to_str(params)
            jobs_params.append((name, params))
    return jobs_params


def params_list_to_str(params):
    if isinstance(params, list):
        params_str = " ".join(str(s) for s in params)
    else:
        params_str = params
    return params_str


def add_mpirun(distributed, amlt_config, gpu, max_sku, cmd, idx):
    if not distributed:
        amlt_config["jobs"][idx]["command"][5] = cmd
        return

    if gpu > 1:
        use_ib = True if "IB" in args.interconnect_type else False
        amlt_config["jobs"][idx]["submit_args"]["env"] = {
            "NCCL_ASYNC_ERROR_HANDLING": 1,
            "NCCL_IB_DISABLE": 0 if use_ib else 1,
            "NCCL_DEBUG": "INFO",
            "MKL_THREADING_LAYER": "GNU",
        }
        if not args.adapt_docker_image_to_singularity:
            amlt_config["jobs"][idx]["submit_args"]["env"]["AMLT_DOCKERFILE_TEMPLATE"] = "default"

        amlt_config["jobs"][idx]["process_count_per_node"] = max_sku if gpu > max_sku else gpu
        amlt_config["jobs"][idx]["mpi"] = True
        amlt_config["jobs"][idx]["command"][5] = cmd
    else:
        amlt_config["jobs"][idx]["command"][5] = cmd


def submit_job(args):
    if args.service == "singularity":
        # Singularity
        max_sku = args.gpus_per_node
        num_gpu = args.num_nodes * args.gpus_per_node
        # CPU job
        if num_gpu == 0:
            # only support 1 CPU job
            sku = f"C1@{args.region}"
        else:
            # GPU job
            sku = f"{args.num_nodes}x{args.memory_size}G{args.gpus_per_node}"
            sku = f"{sku}-{args.gpu_type}"
            if args.interconnect_type.lower() != "empty":
                sku = f"{sku}-{args.interconnect_type}@{args.region}"
    else:
        raise ValueError(f"Do NOT support service {args.service} !")

    if args.exp_name:
        exp_name = f"{args.exp_name}"
    else:
        args.exp_name = "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
        exp_name = f"{args.exp_name}"

    project_path = Path(__file__).parent.parent
    template_config_path = project_path.joinpath(args.template_config)
    with open(template_config_path, "r", encoding="utf-8") as f:
        amlt_config = yaml.safe_load(f)

    amlt_config["description"] = exp_name
    if args.service == "singularity":
        amlt_config["target"]["service"] = "sing"
        amlt_config["target"]["name"] = args.cluster
    else:
        print(f"unsupported service ({args.service})")
        sys.exit(1)

    # Specifying docker information
    amlt_config["environment"]["registry"] = args.image_registry
    amlt_config["environment"]["image"] = f"{args.image_repo}/{args.image_name}"
    if r"docker.io" != args.image_registry:
        docker_registry = f"{args.image_repo}.{args.image_registry}"
        amlt_config["environment"]["registry"] = docker_registry
        amlt_config["environment"]["image"] = args.image_name
        # Set docker username
        amlt_config["environment"]["username"] = args.docker_username
    else:
        raise ValueError("Please use docker image in the Azure Container Registry (ACR).")

    # install mpi4py for distributed training
    if args.distributed:
        amlt_config["environment"]["setup"].append("pip install --user mpi4py")

    # extra configure for TorchTTS
    if args.tool_type.lower() == "torchtts":
        amlt_config["environment"]["setup"].append("pip install --user --no-deps -e .")
    if args.extra_env_setup_cmd is None:
        pass
    elif len(args.extra_env_setup_cmd) > 0:
        amlt_config["environment"]["setup"].append(args.extra_env_setup_cmd)
    # sleeping for 60 seconds to improve stability
    amlt_config["environment"]["setup"].append("sleep 60")

    if args.local_code_dir is not None:
        amlt_config["code"]["local_dir"] = str(args.local_code_dir.resolve())

    amlt_config["storage"]["data_blob"]["storage_account_name"] = get_data_storage_by_region_name(args.region)
    amlt_config["storage"]["output"]["storage_account_name"] = get_model_storage_by_region_name(args.region)
    # amlt_config["storage"]["persistent_blob"]["storage_account_name"] = \
    #     get_persistent_storage_by_region_name(args.region)
    amlt_config["storage"]["data_blob"]["container_name"] = args.data_container_name
    amlt_config["storage"]["output"]["container_name"] = args.model_container_name
    # amlt_config["storage"]["persistent_blob"]["container_name"] = \
    #     args.persistent_container_name

    amlt_model_dir = "$$AMLT_OUTPUT_DIR"
    if args.service == "singularity":
        amlt_log_dir = "$$AMLT_DIRSYNC_DIR"
    else:
        raise ValueError(f"unsupported service {args.service}")

    job_template = copy.deepcopy(amlt_config["jobs"][0])
    jobs_params = parse_extra_params(args.extra_params, exp_name)

    for idx in range(len(jobs_params)):
        job_param = jobs_params[idx]
        job_name, params = job_param

        if idx == 0:
            amlt_config["jobs"][idx] = copy.deepcopy(job_template)
        else:
            amlt_config["jobs"].append(copy.deepcopy(job_template))

        amlt_config["jobs"][idx]["name"] = job_name
        amlt_config["jobs"][idx]["sku"] = sku
        if not args.adapt_docker_image_to_singularity:
            amlt_config["jobs"][idx]["submit_args"]["env"]["AMLT_DOCKERFILE_TEMPLATE"] = "default"

        if args.service == "singularity":
            amlt_config["jobs"][idx]["sla_tier"] = args.sla_tier
            del amlt_config["jobs"][idx]["submit_args"]["container_args"]

        cmd = f"{args.run_cmd} {params}"

        model_arg = None
        if args.tool_type.lower() == "torchtts":
            model_dir = f"trainer.save_path={amlt_model_dir}"
            log_dir = f"hydra.run.dir={amlt_log_dir}"
            cyber_eo_config = f"trainer.model_registry_region={args.region}"
            if args.set_model_dir:
                model_arg = "trainer.save_path"
        elif args.tool_type.lower() == "hydra":
            model_dir = f"trainer.model_dir={amlt_model_dir}"
            log_dir = f"trainer.log_dir={amlt_log_dir} hydra.run.dir=$$AMLT_CODE_DIR"
            cyber_eo_config = f"trainer.model_registry_region={args.region}"
            if args.set_model_dir:
                model_arg = "trainer.model_dir"
        elif args.tool_type.lower() == "default":
            model_dir = f"--model-dir {amlt_model_dir}"
            log_dir = f"--log-dir {amlt_log_dir}"
            cyber_eo_config = f"--model-registry-region {args.region}"
            if args.set_model_dir:
                model_arg = "--model-dir"
        elif args.tool_type.lower() == "customized":
            model_dir_name = args.customized_model_dir_name
            log_dir_name = args.customized_log_dir_name
            model_registry_region_name = args.customized_model_registry_region_name
            if not model_dir_name.endswith("="):
                model_dir_name += " "
            if not log_dir_name.endswith("="):
                log_dir_name += " "
            if not model_registry_region_name.endswith("="):
                model_registry_region_name += " "
            model_dir = f"{model_dir_name}{amlt_model_dir}"
            log_dir = f"{log_dir_name}{amlt_log_dir}"
            cyber_eo_config = f"{model_registry_region_name}{args.region}"
            if args.set_model_dir:
                model_arg = model_dir_name
        else:
            raise ValueError(f"unsupported tool {args.tool_type}")

        if model_arg and model_arg in args.extra_params:
            cmd += f" {log_dir}"
        else:
            cmd += f" {model_dir} {log_dir}"

        if args.enable_cyber_eo:
            cmd += f" {cyber_eo_config}"

        if args.use_dash:
            # dash does not support -o pipefail
            amlt_config["jobs"][idx]["command"][0] = "set -ex"
        else:
            amlt_config["jobs"][idx]["command"][0] = "set -exo pipefail"
        cmd = f"{cmd.strip()} || exit 1"
        add_mpirun(args.distributed, amlt_config, num_gpu, max_sku, cmd, idx)

    random_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
    amlt_cfg_file = Path(__file__).parent.joinpath(f"tmp_cfg_{random_str}.yaml")

    with open(amlt_cfg_file, "w", encoding="utf-8", newline="") as fout:
        yaml.dump(amlt_config, fout)

    if not args.prepare_only:
        # clean up
        atexit.register(cleanup, [amlt_cfg_file])
        cmd = f"amlt run -d {exp_name} {amlt_cfg_file} {exp_name}"
        if args.no_prompts:
            cmd = cmd.replace("amlt run", "amlt run --yes --replace")
        try:
            subprocess.run(cmd, check=True, shell=SHELL)
        except subprocess.CalledProcessError as e:
            config_home = get_config_home()
            print(f"Error: {e}")
            print(
                'INFO: 1 - If you see "The provided target name XXXXXXXX could not be found.", '
                'please run "amlt tl sing -v" manually to refresh the target name and try again.'
            )
            print(
                "INFO: 2 - If you see \"Authorization failed for repository '' on registry 'XXXXXXX.azurecr.io'. "
                "Credentials invalid for username 'XXX'.\", please check if the docker image exists."
            )
            print(
                'INFO: 3 - If you still see "Authorization failed or Credentials invalid", please delete '
                f"{config_home} and try again."
            )
            print(
                "INFO: 4 - If you still face errors and can not fix by yourself, please update the Submitter to the "
                "latest version and try again."
            )
            print(
                "INFO: 5 - If you still face errors after updating the Submitter, please request help via "
                "https://teams.microsoft.com/l/channel/19%3A1f58a1d208024240a372879b347d1cd4%40thread.tacv2/"
                "Support?groupId=24b7534b-7647-4cef-b00f-c3ca5d1812dd&tenantId=72f988bf-86f1-41af-91ab-2d7cd011db47."
            )
            sys.exit(1)
        return exp_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template-config", "-c", type=Path, default="utils/amlt_utils/amlt_cfg.yaml", help="template config file path"
    )
    parser.add_argument(
        "--cluster", "-s", type=str, default="v100-16gb-scus", help="name of cluster (default: 'v100-16gb-scus')"
    )
    parser.add_argument(
        "--virtual-cluster",
        "-vc",
        type=str,
        default="speech-itp-tts",
        help="name of virtual cluster (default: 'speech-itp-tts')",
    )
    parser.add_argument(
        "--image-registry", type=str, default="azurecr.io", help="docker image registry (default: 'azurecr.io')"
    )
    parser.add_argument(
        "--image-repo",
        type=str,
        default="sramdevregistry",
        help="docker image name prefix (default: 'sramdevregistry')",
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default="pytorch:1.13.0-py38-cuda11.6-cudnn8-ubuntu20.04",
        help="docker image name " "(pytorch:1.13.0-py38-cuda11.6-cudnn8-ubuntu20.04)",
    )
    parser.add_argument(
        "--data-container-name",
        type=str,
        default="philly-ipgsp",
        help="Azure Blob storage container name for data " "(default: 'philly-ipgsp')",
    )
    parser.add_argument(
        "--model-container-name",
        type=str,
        default="philly-ipgsp",
        help="Azure Blob storage container name for model " "(default: 'philly-ipgsp')",
    )
    parser.add_argument(
        "--persistent-container-name",
        type=str,
        default="trainingdata",
        help="Azure Blob storage container name for persistent data" "(default: 'trainingdata')",
    )
    parser.add_argument(
        "--distributed", type=str_to_bool, default="false", help="using distributed training (default: false)"
    )
    parser.add_argument(
        "--skip-setup",
        type=str_to_bool,
        default="false",
        help="If specified, will skip automatic setup and " "all setup checks.",
    )
    parser.add_argument(
        "--amlt-project",
        type=str,
        required=False,
        help="If specified, will use it, instead of username " "for PT project name.",
    )
    parser.add_argument(
        "--exp-name",
        "-n",
        type=str,
        default="",
        help="amulet experiment name. If empty, will use "
        "random string. If not empty, will use it as job "
        "name on portal as well",
    )
    parser.add_argument("--run-cmd", type=str, required=True, help="running command")
    parser.add_argument(
        "--extra-params", "-e", type=str, default="", help="extra parameters for the job (default: " ")"
    )
    parser.add_argument(
        "--prepare-only", type=str_to_bool, default="false", help="only prepare repository for job submission"
    )
    parser.add_argument(
        "--service",
        type=str,
        default="singularity",
        choices=["singularity"],
        help="service type (default: singularity)",
    )
    parser.add_argument("--local-code-dir", type=Path, default=None, help="absolute path of local code")
    parser.add_argument(
        "--extra-env-setup-cmd",
        type=str,
        default=None,
        help="extra command to set up running environment (default: None)",
    )
    parser.add_argument(
        "--sla-tier",
        type=str,
        default="Standard",
        choices=["Premium", "Standard", "Basic"],
        help="Service Level Agreement tier for singularity jobs (default: Standard)",
    )
    parser.add_argument(
        "--set-model-dir", type=str_to_bool, default="true", help="set model dir by AMLT (default: true)"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        choices=["eastus", "southcentralus", "westus2", "westus3", "redmond", "rrlab"],
        help="which region to run the job (default: None).",
    )
    parser.add_argument("--num-nodes", type=int, default=1, help="specify the number of nodes to reserve (default: 1)")
    parser.add_argument("--gpus-per-node", type=int, default=4, help="GPUs per node (default: 4)")
    parser.add_argument("--memory-size", type=int, default=16, help="memory of GPUs (default: 16GB)")
    parser.add_argument("--gpu-type", type=str, default="V100", help="GPU type (default: V100)")
    parser.add_argument("--interconnect-type", type=str, default="Empty", help="interconnect type (default: Empty)")
    parser.add_argument(
        "--tool-type",
        type=str,
        default="Default",
        choices=["TorchTTS", "Default", "Hydra", "Customized"],
        help="tool type (default: Default)",
    )
    parser.add_argument(
        "--customized-model-dir-name",
        type=str,
        default="--model-dir",
        help="customized model dir name (default: '--model-dir')",
    )
    parser.add_argument(
        "--customized-log-dir-name",
        type=str,
        default="--log-dir",
        help="customized log dir name (default: '--log-dir')",
    )
    parser.add_argument(
        "--customized-model-registry-region-name",
        type=str,
        default="--model-registry-region",
        help="customized model registry region name (default: '--model-registry-region')",
    )
    parser.add_argument(
        "--use-dash", type=str_to_bool, default="false", help="use dash as default shell (default: false)"
    )
    parser.add_argument(
        "--key-vault-name",
        type=str,
        default="exawatt-philly-ipgsp",
        help="key vault name for azure docker authentication (default: exawatt-philly-ipgsp)",
    )
    parser.add_argument(
        "--docker-username", type=str, default="tts-itp-user", help="docker user name (default: tts-itp-user)"
    )
    parser.add_argument(
        "--adapt-docker-image-to-singularity",
        type=str_to_bool,
        default="true",
        help="Run the singularity installer and validator to adapt docker image for singularity (default: true)",
    )
    parser.add_argument(
        "--enable-cyber-eo",
        type=str_to_bool,
        default="false",
        help="Enable Executive Order (EO) on Improving the Nations Cybersecurity which was issued by the President "
        "Biden on May 12, 2021. (default: false)",
    )
    parser.add_argument(
        "--no-prompts", type=str_to_bool, default="false", help="If specified, will skip prompts from 'amlt run'."
    )

    args = parser.parse_args()

    if args.num_nodes * args.gpus_per_node > 1:
        if not args.distributed:
            raise ValueError("'distributed' must be set as 'True' when training on multi GPUs.")

    docker_registry = f"{args.image_repo}.{args.image_registry}"
    if not args.skip_setup:
        check_setup_amulet(
            args.amlt_project, args.model_container_name, docker_registry, args.docker_username, args.key_vault_name
        )

    submit_job(args)
