#!/bin/bash

# Copyright  2021  Microsoft (author: Ke Wang)

set -euo pipefail

AMLT_VERSION="10.2.1"

# Install AMLT
python -m pip install -U pip
pip install -U amlt==${AMLT_VERSION} \
  --extra-index-url https://msrpypi.azurewebsites.net/stable/leloojoo