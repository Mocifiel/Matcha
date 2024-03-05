#!/bin/bash

# Copyright  2021  Microsoft (author: Ke Wang)

set -euo pipefail

AMLT_VERSION="9.20.1"

# Install AMLT
python -m pip install -U pip
pip install -U amlt==${AMLT_VERSION} \
  --extra-index-url https://msrpypi.azurewebsites.net/stable/leloojoo

# Install ZettaSDK
# https://dev.azure.com/speedme/SpeeDME/_artifacts/feed/ZettASDK
python -m pip install --upgrade pip
python -m pip install keyring artifacts-keyring
python -m pip install azure-core azureml-sdk azure-storage-blob
python -m pip install zettasdk zettasdk-batch \
  --extra-index-url=https://pkgs.dev.azure.com/speedme/SpeeDME/_packaging/ZettASDK%40Release/pypi/simple/