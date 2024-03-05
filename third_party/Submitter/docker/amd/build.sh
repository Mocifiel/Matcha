#!/bin/bash

# Copyright  2022  Microsoft (author: Ke Wang)

set -euo pipefail

registry="singularitybase"
validator_image_repo="validations/base/singularity-tests"
installer_image_repo="installer/base/singularity-installer"
image_name_tag="sramdevregistry.azurecr.io/pytorch:2.0.1-py38-rocm5.4-ubuntu20.04"

az acr login -n $registry

validator_image_tag=$(
  az acr manifest list-metadata \
    --registry $registry \
    --name $validator_image_repo \
    --orderby time_desc \
    --query '[].{Tag:tags[0]}' \
    --output tsv --top 1
)

validator_image="${registry}.azurecr.io/${validator_image_repo}:${validator_image_tag}"

installer_image_tag=$(
  az acr manifest list-metadata \
    --registry $registry \
    --name $installer_image_repo \
    --orderby time_desc \
    --query '[].{Tag:tags[0]}' \
    --output tsv --top 1
)

installer_image="${registry}.azurecr.io/${installer_image_repo}:${installer_image_tag}"

docker_build_cmd=$(cat <<- EOF
docker build -t $image_name_tag . -f Dockerfile
  --build-arg INSTALLER_IMAGE=$installer_image
  --build-arg VALIDATOR_IMAGE=$validator_image
EOF
)
echo $docker_build_cmd

docker build -t $image_name_tag . -f Dockerfile \
  --build-arg INSTALLER_IMAGE=$installer_image \
  --build-arg VALIDATOR_IMAGE=$validator_image \
  --progress=plain
