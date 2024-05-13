# Dockers

Dockers for speech tasks. (Ubuntu + ROCm)

## Requirements

In order to use this image you must have Docker Engine installed. Instructions
for setting up Docker Engine are
[available on the Docker website](https://docs.docker.com/engine/installation/).

## Usage

### Build

```sh
# login
az login --use-device-code
az acr login --name sramdevregistry
# build
bash build.sh
# push to ACR
docker push sramdevregistry.azurecr.io/submitter:pytorch201-py310-rocm57-ubuntu2004
# pull
docker pull sramdevregistry.azurecr.io/submitter:pytorch201-py310-rocm57-ubuntu2004
```

### Running PyTorch scripts

It is possible to run PyTorch programs inside a container using the
`python3` command. For example, if you are within a directory containing
some PyTorch project with entrypoint `main.py`, you could run it with
the following command:

```bash
docker run --rm -it --init \
  -v /mnt/workspace:/mnt/workspace \
  sramdevregistry.azurecr.io/submitter:pytorch201-py310-rocm57-ubuntu2004 bash
```

Here's a description of the Docker command-line options shown above:

* `--rm`: Automatically remove the container when it exits.
* `-i`: Interactive mode.
* `-t`: Allocate a pseudo-TTY.
* `--init`: Run an init inside the container that forwards signals and reaps processes.
* `-v /mnt/workspace:/mnt/workspace`: Mounts /mnt/workspace in local machine into the
  container /mnt/workspace. Optional.
