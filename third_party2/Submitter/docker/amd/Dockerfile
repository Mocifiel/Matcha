# https://singularitydocs.azurewebsites.net/docs/tutorials/custom_images/#singularity-image-validator
ARG BASE_IMAGE="ptebic.azurecr.io/public/azureml/aifx/stable-ubuntu2004-rocm542-py38-torch201"
ARG INSTALLER_IMAGE
ARG VALIDATOR_IMAGE

FROM $BASE_IMAGE as base
FROM $INSTALLER_IMAGE as installer
FROM $VALIDATOR_IMAGE as validator

FROM base

MAINTAINER Ke Wang <wake@microsoft.com>

# Set arguments
ARG TIME_ZONE=Asia/Shanghai
ARG OPENMPI_VERSIONBASE=4.1
ARG OPENMPI_VERSION=${OPENMPI_VERSIONBASE}.5

# Environment variables
# Optionally set validation arguments to run additional checks for Nvidia drivers (AMD or NVIDIA)
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8 \
    SINGULARITY_IMAGE_ACCELERATOR="AMD" \
    DEBIAN_FRONTEND=noninteractive

# Install software needed for the workload (figlet is need for the singularity)
RUN apt-get update && \
    apt-get install --no-install-recommends --no-install-suggests -yq \
        --allow-downgrades --allow-change-held-packages \
    vim git curl locales htop wget sudo tmux \
    zip unzip \
    ffmpeg sox lame libsox-dev libsox-fmt-all \
    libsndfile1 \
    build-essential autotools-dev rsync jq \
    cmake g++ gcc \
    net-tools \
    ca-certificates \
    figlet && \
    locale-gen en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get purge --auto-remove && \
    apt-get clean && \
    figlet Singularity

# Set timezone
RUN apt-get update && \
    apt-get --fix-broken install && \
    apt-get install -y tzdata && \
    ln -sf /usr/share/zoneinfo/${TIME_ZONE} /etc/localtime && \
    echo ${TIME_ZONE} > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

# Optionally set validation arguments to run additional checks for Nvidia drivers (AMD or NVIDIA)
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8 \
    SINGULARITY_IMAGE_ACCELERATOR="AMD"

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v${OPENMPI_VERSIONBASE}/downloads/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar zxf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install pip packages
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
        mkl mkl-include numpy scipy scikit-learn-intelex matplotlib ipython cython \
        pyyaml tqdm requests_ntlm librosa ruamel.yaml pillow \
        soundfile inflect unidecode pysptk museval seaborn onnxruntime \
        h5py sox webdataset tensorboard mpi4py scikit-fuzzy mpi4py

# Change default shell to bash
RUN echo "dash dash/sh boolean false" | debconf-set-selections && \
    DEBIAN_FRONTEND=noninteractive dpkg-reconfigure dash

# Copy files
COPY . /app

# Get the installation scripts
COPY --from=installer /installer /opt/microsoft/_singularity/installations/

# Install components required for running this image in Singularity
RUN /opt/microsoft/_singularity/installations/singularity/installer.sh

# Get the validation scripts
COPY --from=validator /validations /opt/microsoft/_singularity/validations/

# Run the validation
RUN /opt/microsoft/_singularity/validations/validator.sh

# Set working directory
WORKDIR /mnt
RUN chmod -R a+w /mnt
