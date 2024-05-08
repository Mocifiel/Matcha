# Dockers

Dockers for speech tasks. (Ubuntu + ROCm)

## Requirements

In order to use this image you must have Docker Engine installed. Instructions
for setting up Docker Engine are
[available on the Docker website](https://docs.docker.com/engine/installation/).

## Usage

### Build

```sh
# build
bash build.sh
# login (for first time)
docker login
# push to Docker Hub
docker push sramdevregistry.azurecr.io/pytorch:2.0.1-py38-rocm5.4-ubuntu20.04
# pull
docker pull sramdevregistry.azurecr.io/pytorch:2.0.1-py38-rocm5.4-ubuntu20.04
```

### Running PyTorch scripts

It is possible to run PyTorch programs inside a container using the
`python3` command. For example, if you are within a directory containing
some PyTorch project with entrypoint `main.py`, you could run it with
the following command:

```bash
docker run --rm -it --init \
  -v /mnt/workspace:/mnt/workspace \
  sramdevregistry.azurecr.io/pytorch:2.0.1-py38-rocm5.4-ubuntu20.04 bash
```

Here's a description of the Docker command-line options shown above:

* `--rm`: Automatically remove the container when it exits.
* `-i`: Interactive mode.
* `-t`: Allocate a pseudo-TTY.
* `--init`: Run an init inside the container that forwards signals and reaps processes.
* `-v /mnt/workspace:/mnt/workspace`: Mounts /mnt/workspace in local machine into the
  container /mnt/workspace. Optional.
