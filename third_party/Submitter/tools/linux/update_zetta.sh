#!/bin/bash

# Copyright  2021  Microsoft (author: Ke Wang)

set -euo pipefail

# Install ZettaSDK
# https://dev.azure.com/speedme/SpeeDME/_artifacts/feed/ZettASDK
python -m pip install --upgrade pip
python -m pip install keyring artifacts-keyring
python -m pip install azure-core azureml-sdk azure-storage-blob
python -m pip install zettasdk zettasdk-batch \
  --extra-index-url=https://pkgs.dev.azure.com/speedme/SpeeDME/_packaging/speedme-upstream/pypi/simple/