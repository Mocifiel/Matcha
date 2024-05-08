import argparse
import logging
import os
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)20s\t%(levelname)s\tP%(process)d\t%(message)s",
)


def run_subprocess(command, arguments, cwd=None):
    """Run a subprocess command in VM code started with `ParallelRunStep` (PRS) or `SpeechOptimizedParallelRunStep`.
    The PRS implementation does not normally show subprocess output in the user log files. Instead they are captured
    into the sys/node log files, which many users may not know how to find. This function runs the given command
    with the given arguments and captures stdout/stderr output into the logging stream, which puts timestamps on the
    entries as well as putting them into the user log file.

    Parameters
    ----------
    command : str
        Command to run.
    arguments: list[str] | None
        List containing arguments to pass to `command`. Can be None.
    cwd: str | None
        Sets the current directory before the subprocess is executed.

    Raises
    ------
    subprocess.CalledProcessError
        Raised when the invoked process exits with a non-zero exit code.
    """  # noqa: E501
    LOG = logging.getLogger("zetta.subprocess")
    command_list = [command]
    if arguments:
        if not isinstance(arguments, list):
            raise ValueError("arguments parameter must be a list")

        command_list.extend(arguments)

    if cwd:
        LOG.info("Using working directory %s ...", cwd)
    LOG.info("Executing %s ...", " ".join(command_list))

    proc = subprocess.Popen(
        command_list, encoding="utf8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd
    )

    while proc.poll() is None:
        LOG.info(proc.stdout.readline().rstrip())
    lines = proc.stdout.readlines()
    for line in lines:
        LOG.info(line.rstrip())

    if proc.returncode != 0:
        LOG.error("Processed returned exit code %s", proc.returncode)
        raise subprocess.CalledProcessError(proc.returncode, " ".join(command_list))


def main():
    LOG = logging.getLogger("zetta")
    # ------------------------------------------------------------------------
    # Mount Standard Azure Blob
    # ------------------------------------------------------------------------
    standard_mount_path = "/datablob"
    LOG.info(f"Mounting {args.standard_datastore_name} to {standard_mount_path} ...")
    run_subprocess(
        "ln",
        [
            "-Ts",
            os.environ[f"AZUREML_DATAREFERENCE_{args.standard_datastore_name}"],
            standard_mount_path,
        ],
    )

    # ------------------------------------------------------------------------
    # Mount Premium Azure Blob
    # ------------------------------------------------------------------------
    premium_mount_path = "/modelblob"
    LOG.info(f"Mounting {args.premium_datastore_name} to {premium_mount_path} ...")
    run_subprocess(
        "ln",
        [
            "-Ts",
            os.environ[f"AZUREML_DATAREFERENCE_{args.premium_datastore_name}"],
            premium_mount_path,
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--standard-datastore-name", type=str, required=True)
    parser.add_argument("--premium-datastore-name", type=str, required=True)
    args = parser.parse_args()

    main()
