# Submitter

## FAQ

> If you do not find the solution to solve your issue, please raise your question in
> [Submitter Support Teams Channel](https://teams.microsoft.com/l/team/19%3aRd745LI737K4Hbwiuwqf3beR7-8pMxgpECe7wvqf7tc1%40thread.tacv2/conversations?groupId=24b7534b-7647-4cef-b00f-c3ca5d1812dd&tenantId=72f988bf-86f1-41af-91ab-2d7cd011db47).

1. I setup the environment failed. How Can I fix it?

    - If you face the permission issue or the environment can't be set up successfully. Please delete the follow folder and try again.

        - Windows: `C:\Users\[alias]\AppData\Roaming\amulet`
        - Linux: `$HOME/.config/amulet`

2. How to run the shell on the Windows?

    - We strongly suggest you to use `Git-Bash`, whose path is `[INSTALL_FOLDER]\bin\bash.exe`. `CMD` and `PowerShell` are not recommended.
    If you use these terminal, you may need pay more effort to set up the environment.

3. Which version of the Python should I use?

    - Currently, we suggest you to use **Python 3.9**.

4. For other questions, please find the solution in [Know issue](#known-issues).

    - If you still can not solve the issue, please raise the issue in [Submitter Support Teams Channel](https://teams.microsoft.com/l/team/19%3aRd745LI737K4Hbwiuwqf3beR7-8pMxgpECe7wvqf7tc1%40thread.tacv2/conversations?groupId=24b7534b-7647-4cef-b00f-c3ca5d1812dd&tenantId=72f988bf-86f1-41af-91ab-2d7cd011db47).

5. How to enable AzureML Metrics Logging

    From Amulet 9.21.0, it stopped automatically call `mlflow.autolog()` anymore. Please refer to the following code to
    continue logging metrics to MLflow. Make sure you `pip install mlflow azureml-mlflow`.

    ```python
    def get_rank():
       if "RANK" in os.environ:
           return int(os.environ["RANK"])
       if "OMPI_COMM_WORLD_RANK" in os.environ:
           return int(os.environ["OMPI_COMM_WORLD_RANK"])
       return 0

    import mlflow

    rank = get_rank()
    run_id = os.environ.get("MLFLOW_RUN_ID", None)
    if rank == 0 and run_id is not None:
        mlflow.autolog()
        mlflow.start_run(os.environ.get("MLFLOW_RUN_ID"))
        try:
            # pytorch autologging is not enabled by default
            mlflow.pytorch.autolog(log_every_n_step=int(os.environ.get("AMLT_LOG_EVERY_N_STEP", 1)))
        except Exception:
            pass

    train()  # your training code is here

    if rank == 0 and run_id is not None:
        mlflow.end_run()
    ```

## Introduction

This is a repertory for Microsoft TTS Team GPU job submission. Following the example,
you can submit your job to the **Singularity** using
[Amulet](https://amulet-docs.azurewebsites.net/main/setup.html), a.k.a., AMLT.

| System       | Python 3.8.13 | Python 3.9.12 |
|:------------:|:-------------:|:-------------:|
| Ubuntu 18.04 |   pass        |   pass        |
| Window 11    |   pass        |   pass        |

***During training, standard Azure blob will be mounted to `/datablob` and premium Azure blob will be mounted to `/modelblob` by default. Specifically, data and cluster used for training should be in the same region in order to save the data reading time.***
For example, if you submit a job to south central US (scus),
you should upload your data into `stdstoragetts01scus` and the models will be saved to `exawattaiprmbtts01scus`.
For more information about storage, please refer to [Storage Doc](storage.md).

|Storage Type|    Storage Account   | Containers |Folder Name|    Mount Path   |
|:----------:|:--------------------:|:----------:|:---------:|:---------------:|
|Premium     |exawattaiprmbtts01scus|philly-ipgsp|    wake   |`/modelblob/wake`|
|Premium     |exawattaiprmbtts01wus2|philly-ipgsp|    wake   |`/modelblob/wake`|
|Premium     |exawattaiprmbtts01eus |philly-ipgsp|    wake   |`/modelblob/wake`|
|Standard    |stdstoragetts01scus   |philly-ipgsp|    wake   |`/datablob/wake` |
|Standard    |stdstoragetts01wus2   |philly-ipgsp|    wake   |`/datablob/wake` |
|Standard    |stdstoragetts01eus    |philly-ipgsp|    wake   |`/datablob/wake` |

In November 2022, ITP was migrated to Singularity. You can use the following command to check the clusters and quotas you can use.

```bash
# Query the available clusters
amlt tl sing -v  # or amlt target list sing -v

# Query the quotas
amlt ti sing -v # or amlt target info sing -v

# Query the instance series details
amlt cache instance-type -s <series>  # amlt cache instance-type -s NCv3
```

![Singularity](figure/sing_clusters_quotas.png)

**IMPORTANT:** For your training recipe, please add the following code to save your models and tensorboard files. If you use the following code to write tensorboard file to `log_dir`,
you can use the tensorboard services provided by ITP team.
**Moreover, you do NOT need to set `--model-dir` and `--log-dir` by yourself when you use `utils/amlt_submit.py` to submit job to ITP. The two parameters will be automatically set by `utils/amlt_submit.py`.**

```python
parser.add_argument('--model-dir', type=Path, required=True, help='model directory')
parser.add_argument('--log-dir', type=Path, default=None, help='log directory')

args = parser.parse_args()
# create log directory to save tensorboard
# summary written in the default directory can be shown on ITP Portal
if args.log_dir is not None:
    log_dir = args.log_dir
else:
    log_dir = model_dir.joinpath('log')
log_dir.mkdir(parents=True, exist_ok=True)
args.model_dir.mkdir(parents=True, exist_ok=True)
```

You can also disable the setting and write log files and models to anywhere you want by setting `--set-model-dir` as `"false"` for `utils/amlt_submit.py`,
but this setting is **NOT** recommended.

In order to write safe shell script and make the status of job displayed in ITP portal correctly, `set -euo pipefail` is enable by default.
But the default shell of Ubuntu is dash which does not support `-o pipefail`.
Thus, please add the following configures in your docker file.

```bash
# Change default shell to bash
RUN echo "dash dash/sh boolean false" | debconf-set-selections && \
    DEBIAN_FRONTEND=noninteractive dpkg-reconfigure dash
```

> set -o pipefail causes a pipeline (for example, curl -s https://sipb.mit.edu/ | grep foo)
> to produce a failure return code if any command errors.
> Normally, pipelines only return a failure if the last command errors.
> In combination with set -e, this will make your script exit if any command in a pipeline errors.

You can disable the `-o pipefail` feature if you'd like to use dash instead of bash by setting `--use-dash` as `true` for `utils/amlt_submit.py`, but this setting is **NOT** recommended.

## Running examples

1. [**IMPORTANT**] Before submitting job using AMLT, please create a file named `.amltignore` in your project root folder.
This file is very similar with `.gitignore` and will ignore some files which will not be used
during training or inference. I also provided an example, please find it [here](.amltignore).
For more information about AMLT, please refer to [Amulet](https://amulet-docs.azurewebsites.net/master/setup.html).

2. Note that when you first try to launch a job with `utils/amlt_submit.py`, it will need to perform some setup on your machine
(install a few python packages and initialize Amulet project). You will be asked if you want to perform the setup automatically
(usually you should agree). During the setup process, you will see a web link and verification code on your command window.
Open the link in any web browser and input the verification code.

### Examples to submit job

- Submit training job

    **Before submitting job, please join the SGs group named "speech-itp-tts-usr" on [IDWeb](https://idweb/identitymanagement/default.aspx).
    Otherwise, you will have no permission to access the resources.
    If you are a vendor or an intern, your also need to join he SGs group named
    "exawatt-users" (FTEs do NOT need to join).**

    ```bash
    # On Windows, you can use git bash (e.g., C:\Program Files\Git\bin\bash.exe) to run the following command.
    bash jobs/submit_train_sing_amd.sh   # More details, please see the comments.
    ```

    After running this scrips, your code will be uploaded to standard AzureBlob
    (`stdstoragetts01eus/philly-ipgsp/{username}/{project_name}/amlt-code/{random_string}`)
    and training job will be submitted to the GPU cluster. The following image shows the folder structure in Microsoft Azure Storage Explorer.
    The models are in `amlt-results` according your cluster's location, `logs` is used to store log files (e.g., tensorboard) by default.

    ![AML Job](figure/amlt_job.png)


- Submit jobs with different parameters using the same code

    If you want to use the same code to run many jobs with different parameters, you can use the
    [Job List Feature](https://amulet-docs.azurewebsites.net/master/own_jobs.html#define-a-list-of-jobs)
    in Amulet to avoid to upload the same code too many times. We also provide an example
    [script](jobs/submit_train_sing_amd_multi.sh) to use this feature. The format of the parameters likes the following:

    ```bash
    jobname1&&&--data data1;;;jobname2&&&--data data2;;;jobname2&&&--data data3
    ```

    We use `;;;` to separate the parameters for each job and `&&&` is used to separate the job name and parameters for a particular job.

## How to use the tools in your personal project

1. Add Submitter as a submodule for your personal project

    ```bash
    mkdir third_party
    cd third_party
    git submodule add git@ssh.dev.azure.com:v3/wangkenpu/Philly/Submitter
    ```

2. (Optional) Adding custom Azure blob

    Some users in our team maybe will use custom Azure blob (e.g., personal blob or blob from ASR team).
    You can add your storage configure to our [template YAML file](utils/pt/pt_cfg.yaml).
    For more information, you can refer to [Amulet Doc](https://amulet-docs.azurewebsites.net/main/advanced/61_multi_storage_configuration.html).
    The following is an example:

    ```YAML
    storage:
      data_blob:
        storage_account_name: stdstoragetts01scus
        container_name: philly-ipgsp
        mount_dir: /datablob
        mount_options: ["--file-cache-timeout-in-seconds=10000000"]
      output:
        storage_account_name: exawattaiprmbtts01scus
        container_name: philly-ipgsp
        is_output: True                   # optimizes blob mount for write access
        mount_dir: /modelblob
      shared_datastore:
        storage_account_name: teamstorage # your storage account name
        container_name: ptdemo            # container name
        is_output: True                   # if output is true
                                          # else set as false and set configures for mount_options
        mount_dir: /sharedblob            # mount point
    ```

3. (Optional) Using docker in Azure

    You can follow the [example script](jobs/submit_train_amlk8s_azure_docker.sh) to use your docker in Azure.
    The most parts are the same with the regular job, except the following:

    ```bash
    --image-registry "azurecr.io" --image-repo "avenadev" \
    --key-vault-name "avenavault" --docker-username "avenadev-itp" \
    ```

    `avenadev` is your container registry name, `avenavault` is the key vault name to store the secrets.
    In order to authenticate and pull docker successfully on ITP, you should create two secrets using Azure key vault and name
    them as `{container_registry_name}-registry-username` and `{container_registry_name}-registry-pwd` to store the username and
    password like the following:
    ![Key Vault](figure/key_vault.png)

    **If you store the docker image to `sramdevregistry`, please set the `key-vault-name` as `exawatt-philly-ipgsp`.
    We have already stored the key vault to `exawatt-philly-ipgsp` and you don't need to create a new one again.**
    Thus, for our team default Azure Container Registry, you can use the following setting:

    ```bash
    --image-registry "azurecr.io" --image-repo "sramdevregistry" \
    --key-vault-name "exawatt-philly-ipgsp" --docker-username "tts-itp-user" \
    ```

4. Singularity container images

    From December 2022, we have changed form ITP to Singularity. If you want to use custom docker image,
    there are some [requirements](https://singularitydocs.azurewebsites.net/docs/tutorials/custom_images/#requirements-for-custom-image-support) MUST be observed.
    Fortunately, we have provided [AMD](docker/amd/) and [Nvidia](docker/nvidia/) examples to show how to build the custom docker images from scratch.

## Known issues

1. subscription

    > You are currently logged-in to 72f988bf-86f1-41af-91ab-2d7cd011db47 tenant. > > You don't have access to 48b6cd5e-3ffe-4c2e-9e99-5760a42cd093 subscription,
    > please check if it is in this tenant. All the subscriptions that you have access to in this tenant are =...

    You need to delete the `~/.azure*`, or use python script to update your tenant id:

    ``` python
    from azureml.core import Workspace
    from azureml.core.authentication import InteractiveLoginAuthentication

    interactive_auth = InteractiveLoginAuthentication(tenant_id="72f988bf-86f1-41af-91ab-2d7cd011db47", force=True)

    ws = Workspace(subscription_id="48b6cd5e-3ffe-4c2e-9e99-5760a42cd093",
            resource_group="speech-itp-tts",
            workspace_name="speech-itp-tts-ws01-scus",
            auth=interactive_auth)

    print(ws)
    ```

2. Submitting jobs from Linux Non-Managed Devices

    Those of you who can't submit jobs from a non-managed device (such an azure vm or gcr linux machine not in corpnet), try the following:

    Join the Active Linux Users group following the description at this link:
    [Remote access on Linux devices](https://microsoft.sharepoint.com/sites/Network_Connectivity/SitePages/RemoteAccess/Remote-access-on-Linux-devices.aspx).
    Once that is done, wait for a bit, then try pt commands again.
    If that still doesn't work, re-install azure-cli and type az login, manually enter email and use phone auth (instead of cert, current user, etc..).

3. Cannot uninstall 'ruamel-yaml'

    > ERROR: Cannot uninstall 'ruamel-yaml'. It is a distutils installed project and > thus we cannot accurately determine which files belong to it which would lead > to only a partial uninstall.

    1. check the installed path of ruamel

        ```bash
        pip -V
        # output
        # pip 21.0.1 from /data/wake/tools/miniconda3/lib/python3.8/site-packages/pip (python 3.8)
        ```

    2. delete installed ruamel

        ```bash
        # remember to change to your path
        rm -rf /data/wake/tools/miniconda3/lib/python3.8/site-packages/ruamel*
        ```

    Then, re-submit your job.

4. target name not found

    > CC The provided target name "itp-v100-scus" could not be found.
    > EE Check "pt target list" for valid values.
    > EE You can add targets via "pt target add" or "pt target add-defaults".

    This is because you use a new cluster (or AML compute) named `itp-v100-scus` that you never used before.
    Thus you need to refresh your computing target list. Please use the following command to refresh the target list.

    ```bash
    # refresh singularity compute resource
    amlt targe list sing -v
    ```

    Then, you can re-submit your job successfully.

5. Cannot find docker username for ZettA job

    ```bash
    zetta.pipeline_main     INFO    Setting up Docker image configuration ...
    Traceback (most recent call last):
      File "third_party/Submitter/utils/zetta_submit.py", line 167, in <module>
      main()
    File "third_party/Submitter/utils/zetta_submit.py", line 109, in main
      raise ValueError(f"Cannot find docker username for {registry_name}, please run again.")
    ValueError: Cannot find docker username for sramdevregistry, please run again.
    ```

    Currently, ZettA job relies on the certification file which is generated by the AMLT.
    Thus, if you see above error messages, please run the sample job to generate the required certification file. For example:

    ```bash
    bash jobs/submit_train_sing_nvidia.sh
    ```

    **Note: Please remember to cancel your job after you generate the credential file.**

## Note

### Links

- [Amulet](https://amulet-docs.azurewebsites.net/main/index.html)
- [Amulet support](https://teams.microsoft.com/l/channel/19%3ae8a759df27d5407fb9b01d675c8a4832%40thread.skype/Support?groupId=e2708641-ae8b-48b0-9049-1c75809b908f&tenantId=72f988bf-86f1-41af-91ab-2d7cd011db47)
- [Singularity](https://singularitydocs.azurewebsites.net/docs/overview/concepts/)
- [Virtual Clusters](https://ml.azure.com/virtualClusters?flight=itpmerge&tid=72f988bf-86f1-41af-91ab-2d7cd011db47)
- [Singularity IcM Ticket](https://portal.microsofticm.com/imp/v3/incidents/create?tmpl=H3nD93)
- [ZettA Doc](https://speedmedocs.azurewebsites.net/index.html)

## Update tools

```bash
bash tools/linux/update_amlt.sh
bash tools/linux/update_zetta.sh
```
