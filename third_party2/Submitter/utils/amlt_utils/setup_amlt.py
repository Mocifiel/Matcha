#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

import os
import platform
import shutil
import subprocess
import sys
import importlib


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.amlt_utils.skumanager import get_amlt_project_code_storage  # noqa: E402

STORAGE_ACCOUNTS = ["exawattaiprmbtts01wus2", "stdstoragetts01wus2",
                    "exawattaiprmbtts01scus", "stdstoragetts01scus",
                    "exawattaiprmbtts01eus", "stdstoragetts01eus"]

# registry name: [username, keyvault name]
DOCKER_REGISTRY_INFO = {
    "sramdevregistry":    ["tts-itp-user", "exawatt-philly-ipgsp"],
    "azurespeechdockers": ["default-pull", "exawatt-philly-ipgsp"],
    # "avenadev":           ["avenadev-itp", "avenavault"],
}
STORAGE_KEYVAULT_NAME = "exawatt-philly-ipgsp"

AMLT_VERSION = "9.20.1"
AMLT_INSTALL_CMD = (
    "python -m pip install -U pip && "
    f"pip install -U amlt=={AMLT_VERSION} "
    "--extra-index-url https://msrpypi.azurewebsites.net/stable/leloojoo"
)
PIP_UPDATE_CMD = "python -m pip install --upgrade pip"

AMLT_RESULTS_DIR = "amlt-exp-results"
AMLT_PROJ_ACCOUNT = "stdstoragetts01eus"
AMLT_PROJ_CONTAINER = "philly-ipgps"
DOCKER_REGISTRY = "username.docker.io"
SHELL = False if platform.system() == "Windows" else True


def get_authentication(mode: str = None, return_credential: bool = True):
    """Get the Azure authentication credentials.

    Args:
        mode (str, optional): You can specify "device" mode or "SP" mode to
            choose auth type.
            If "mode" is not explicitly specified, we will check if SP
            environment variables are defined and use SP in that case.
            Otherwise, will fall back to DeviceCode authentication.
            Defaults to None.
        return_credential (bool, optional): if True, will return
            credentials object. Otherwise, will return authentication object.
    """
    if mode is not None:
        mode = mode.lower()

    tenant_id = os.environ.get("SERVICE_PRINCIPAL_TENANT_ID")
    service_principal_id = os.environ.get("SERVICE_PRINCIPAL_ID")
    service_principal_password = os.environ.get("SERVICE_PRINCIPAL_PASSWORD")
    if mode is None:
        if all([tenant_id, service_principal_id, service_principal_password]):
            mode = "sp"
        else:
            mode = "device"
    if mode == "sp" and not all([tenant_id, service_principal_id, service_principal_password]):
        raise ValueError("Cannot use SP authentication if SP environment variables are not defined")
    if mode == "sp":
        if return_credential:
            from azure.identity import ClientSecretCredential

            return ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=service_principal_id,
                client_secret=service_principal_password,
            )
        from azureml.core.authentication import ServicePrincipalAuthentication

        auth = ServicePrincipalAuthentication(
            tenant_id=tenant_id,
            service_principal_id=service_principal_id,
            service_principal_password=service_principal_password,
        )
        auth.get_authentication_header()

        return auth
    if mode == "device":
        if return_credential:
            from azure.identity import (
                AzureCliCredential,
                DeviceCodeCredential,
                ChainedTokenCredential,
            )

            try:
                from azure.identity._constants import AZURE_CLI_CLIENT_ID
            except ImportError:
                from azure.identity._constants import (
                    DEVELOPER_SIGN_ON_CLIENT_ID as AZURE_CLI_CLIENT_ID,
                )

            credential = ChainedTokenCredential(
                AzureCliCredential(), DeviceCodeCredential(AZURE_CLI_CLIENT_ID)
            )

            return credential
        return None

    raise ValueError(f"Unknown mode '{mode}'")


def check_python_version(verbose=True):
    if verbose:
        print("\n ********* Checking python version ********* \n")
    if sys.version_info.major != 3:
        raise AssertionError(
            f"Your python version is {platform.python_version()}. "
            f"Please use python3 as default (should lower than python39).")
    else:
        if sys.version_info.minor >= 10:
            raise AssertionError(
                f"Your python version is {platform.python_version()}. "
                f"Do NOT test the tool on python3.{platform.python_version()}. "
                f"Please use the lower version.")
        print(f"Python version is {platform.python_version()}")


def update_pip(verbose=True):
    if verbose:
        print("\n ********* Updating pip ********* \n")
        subprocess.check_call(f"{PIP_UPDATE_CMD}", shell=SHELL)
    else:
        subprocess.check_output(f"{PIP_UPDATE_CMD}", shell=SHELL)


def install_package(package, extra_args=[], verbose=True):
    if verbose:
        print(f"\n ********* Installing python package: {package} ********* \n")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] +
                              extra_args + [package])
    else:
        subprocess.check_output([sys.executable, "-m", "pip", "install"] +
                                extra_args + [package])


def change_command_to_list(command):
    if isinstance(command, str):
        return [x.strip() for x in command.strip().split("&&")]
    else:
        return command


def install_amulet(verbose=True):
    command_list = change_command_to_list(AMLT_INSTALL_CMD)
    if verbose:
        print("\n ********* Installing Amulet ********* \n")
        for command in command_list:
            subprocess.check_call(f"{command}", shell=SHELL)
    else:
        for command in command_list:
            subprocess.check_output(f"{command}", shell=SHELL)


def delete_vault_file():
    if platform.system() == "Windows":
        config_home = os.path.join(os.environ["APPDATA"], "amulet")
    elif platform.system() == "Linux":
        import xdg
        config_home = os.path.join(xdg.XDG_CONFIG_HOME, "amulet")
    else:
        raise ValueError(f"Unsupported system {platform.system()}")
    if os.path.isdir(config_home):
        print(f">>> Remove {config_home} ...")
        shutil.rmtree(config_home)


def prepare_secrets(microsoft_username):
    """Will retrieve all secrets (storage keys, docker password, etc.).

    Will save the information to the local amlt cache.
    """
    # Delete the vault file
    delete_vault_file()

    print("\n ********* Saving Azure keys for Amulet ********* \n")
    from amlt.vault import Vault

    for docker_registry, (docker_username, key_vault_name) in DOCKER_REGISTRY_INFO.items():
        docker_pwd_info = (
            f"amlt cred cr set {docker_registry}/{docker_username} --keyvault-ident "
            f"https://{key_vault_name}.vault.azure.net/secrets/{docker_registry}-registry-pwd"
        )
        subprocess.run(docker_pwd_info, shell=True, check=True)

    for storage_account in STORAGE_ACCOUNTS:
        storage_info = (
            f"amlt cred storage set {storage_account} --keyvault-ident "
            f"https://{STORAGE_KEYVAULT_NAME}.vault.azure.net/secrets/{storage_account}-key"
        )
        subprocess.run(storage_info, shell=True, check=True)

    if Vault().has_passpy():
        store_fn = Vault().set_password_gpg
    else:
        store_fn = Vault().set_password_cleartext

    store_fn("microsoft_username", microsoft_username, realm="meta_info")

    # this is a bit hacky, but will make sure amlt won"t ask for username
    from amlt.globals import CONFIG_HOME

    username_file = os.path.join(CONFIG_HOME, "philly_user")
    with open(username_file, "wt", encoding="utf-8") as fout:
        fout.write(microsoft_username)


def check_secrets(docker_username):
    """Checks if all the secrets are present in amlt cache."""
    try:
        import eventlet
    except ImportError:
        install_package("eventlet", verbose=False)
        import eventlet
        importlib.reload(eventlet)
    eventlet.monkey_patch()

    from amlt.vault import PasswordStorage, Credentials
    from amlt.globals import CONFIG_HOME

    if r".docker.io" not in DOCKER_REGISTRY:
        if not os.path.exists(os.path.join(CONFIG_HOME, "philly_user")):
            return None

        docker_password = None
        with eventlet.Timeout(5, False):
            try:
                docker_password = Credentials.get_container_registry_password(DOCKER_REGISTRY, docker_username)
            except Exception:
                pass
        if not docker_password:
            return None

    for storage_account in STORAGE_ACCOUNTS:
        account_key = Credentials.get_azure_storage_secret(storage_account)
        if not account_key:
            return None
    microsoft_username = PasswordStorage().retrieve_noninteractive(
        "microsoft_username", realm="meta_info", secret_type="Password")
    return microsoft_username


def _check_amulet(amlt_project_name, docker_username):
    # checking that Amulet is installed
    import amlt
    # to pick up a version if was reinstalled
    importlib.reload(amlt)
    if amlt.__version__ != AMLT_VERSION:
        raise ImportError()
    # checking that all required secrets are stored
    username = check_secrets(docker_username)
    if not username:
        raise ImportError()
    # checking that amlt project is set up correctly
    if amlt_project_name:
        project_name = amlt_project_name
    else:
        project_name = username
    try:
        checkout_or_create_project(project_name, username)
    except subprocess.CalledProcessError:
        raise ImportError()
    return username


def checkout_or_create_project(project_name, username):
    """Will try to checkout amlt project or create it if it does not exist."""
    try:
        # trying to switch to project, assuming it exists
        subprocess.run(
            f"amlt project checkout {project_name} "
            f"{AMLT_PROJ_ACCOUNT} {AMLT_PROJ_CONTAINER} {username}",
            check=True,
            shell=SHELL,
        )
    except subprocess.CalledProcessError:
        # trying to create a project
        subprocess.run(
            f"amlt project create {project_name} "
            f"-f {{experiment_name}}/{{job_id}} "
            f"{AMLT_PROJ_ACCOUNT} {AMLT_PROJ_CONTAINER} {username}",
            check=True,
            shell=SHELL,
        )


def check_setup_amulet(amlt_project_name=None, container_name="philly-ipgsp", docker_registry="username.docker.io",
                       docker_username="username", key_vault_name="exawatt-philly-ipgsp"):
    global AMLT_PROJ_ACCOUNT
    AMLT_PROJ_ACCOUNT = get_amlt_project_code_storage()
    global AMLT_PROJ_CONTAINER
    AMLT_PROJ_CONTAINER = container_name
    global DOCKER_REGISTRY
    DOCKER_REGISTRY = docker_registry

    registry_name = docker_registry.strip().split(r".")[0]
    # Check the consistency of the arguments
    if registry_name not in DOCKER_REGISTRY_INFO:
        raise ValueError(f"Unknown docker registry {docker_registry}")
    if docker_username != DOCKER_REGISTRY_INFO[registry_name][0]:
        raise ValueError(f"docker_username should be {DOCKER_REGISTRY_INFO[registry_name][0]}")
    if key_vault_name != DOCKER_REGISTRY_INFO[registry_name][1]:
        raise ValueError(f"key_vault_name should be {DOCKER_REGISTRY_INFO[registry_name][1]}")

    # checking if machine is set up properly
    try:
        return _check_amulet(amlt_project_name, docker_username)
    except ImportError:
        print("Your machine has not been set up correctly to use submission scripts.")
        # check python version and update pip
        check_python_version(verbose=True)
        update_pip(verbose=True)
        try:
            import click
        except ImportError:
            install_package("click==8.1.3", verbose=False)
            import click
        if not click.confirm(
                    "Do you want to perform automatic setup? It will "
                    "install several python packages and set up Amulet "
                    "for you.\n"
                    "During this process you might be asked to "
                    "authenticate in the browser several times and enter your "
                    "Microsoft username. Please provide all required "
                    "information to ensure successful setup",
                    default=True,
                ):
            print("In this case, please follow steps described in "
                  "https://torchspeech.azurewebsites.net/start_training_in_torchspeech/faq.html "
                  "to set up everything manually")
            exit(1)
        print("\n ********* Uninstalling python packages ********* \n")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "amlt", "-y"])
        install_package("azure-identity")
        install_package("azure-keyvault-secrets")
        install_package("pyyaml")
        if os.name == "nt":
            install_package("pywin32==227")
        install_amulet()

        import amlt
        importlib.reload(amlt)

        print("")
        # checking if need to prepare Azure keys or they were already prepared
        microsoft_username = check_secrets(docker_username)
        if not microsoft_username:
            microsoft_username = click.prompt("Enter your Microsoft alias (without domain)")
            prepare_secrets(microsoft_username)
        print("\n ********* Creating amlt project ********* \n")
        if amlt_project_name:
            project_name = amlt_project_name
        else:
            project_name = microsoft_username
        try:
            checkout_or_create_project(project_name, microsoft_username)
        except subprocess.CalledProcessError:
            print("Automatic setup failed with error. Please retry it one more time.")
            exit(1)
        # set default target
        subprocess.run("amlt target list sing -v", check=True, shell=SHELL)

    try:
        microsoft_username = _check_amulet(amlt_project_name, docker_username)
    except ImportError:
        print("Automatic setup failed. Please retry it one more time.")
        exit(1)

    print("\n ********* Setup successfully finished! ********* \n")
    print(f"Note that we created {AMLT_RESULTS_DIR} folder for you. Don't modify "
          f"anything inside that folder manually. You will be able to see "
          f"results of your experiments there.\n")
    return microsoft_username
