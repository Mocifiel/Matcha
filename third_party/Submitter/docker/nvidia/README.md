# Dockers

Dockers for speech tasks. (Ubuntu + CUDA)

## Requirements

In order to use this image you must have Docker Engine installed. Instructions
for setting up Docker Engine are
[available on the Docker website](https://docs.docker.com/engine/installation/).

### CUDA requirements

If you have a CUDA-compatible NVIDIA graphics card, you can use a CUDA-enabled
version of the image to enable hardware acceleration.

You will also need to install `nvidia-docker2` to enable GPU device access
within Docker containers. This can be found at
[NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

## Usage

### Build

```sh
# build
docker build -t sramdevregistry.azurecr.io/pytorch:2.0.1-py39-cuda11.7-ubuntu20.04 .
# login (for first time)
docker login
# push to Docker Hub
docker push sramdevregistry.azurecr.io/pytorch:2.0.1-py39-cuda11.7-ubuntu20.04
# pull
docker pull sramdevregistry.azurecr.io/pytorch:2.0.1-py39-cuda11.7-ubuntu20.04
```

### Running PyTorch scripts

It is possible to run PyTorch programs inside a container using the
`python3` command. For example, if you are within a directory containing
some PyTorch project with entrypoint `main.py`, you could run it with
the following command:

```bash
docker run --rm -it --init \
  --runtime=nvidia \
  --ipc=host \
  -v /mnt/workspace:/mnt/workspace \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  sramdevregistry.azurecr.io/pytorch:2.0.1-py39-cuda11.7-ubuntu20.04 bash
```

Here's a description of the Docker command-line options shown above:

* `--rm`: Automatically remove the container when it exits.
* `-i`: Interactive mode.
* `-t`: Allocate a pseudo-TTY.
* `--init`: Run an init inside the container that forwards signals and reaps processes.
* `--runtime=nvidia`: Required if using CUDA, optional otherwise. Passes the
  graphics card from the host to the container.
* `--ipc=host`: Required if using multiprocessing, as explained at
  https://github.com/pytorch/pytorch#docker-image.
* `-v /mnt/workspace:/mnt/workspace`: Mounts /mnt/workspace in local machine into the
  container /mnt/workspace. Optional.
* `-e NVIDIA_VISIBLE_DEVICES=0`: Sets an environment variable to restrict which
  graphics cards are seen by programs running inside the container. Set to `all`
  to enable all cards. Optional, defaults to all.
