#!/bin/bash

# Copyright  2022  Microsoft (author: Ke Wang)

set -euo pipefail

workspace_name="zetta-amprod-ws01-scus"  # zetta-amprod-ws01-scus, zetta-prod-ws02-eus2, zetta-prod-ws01-wus2, zetta-prod-ws03-wus2
compute_target="ZettA-AML-Target"

command="df -h && "
command=${command}"pwd && "
command=${command}"ls -al / && "
command=${command}"ls -al /datablob/wake && "
command=${command}"ls -al /modelblob/wake && "
command=${command}"touch /datablob/wake/zetta_test.txt && "
command=${command}"rm -f /datablob/wake/zetta_test.txt && "
command=${command}"python --version && "
command=${command}"python -c \"import torch; print(torch.__version__)\""

experiment_name="zetta_test"    # project name (e.g., tacotron/fastspeech)
display_name="command_running"  # experimental name (e.g., Evan/Guy/Aria)

python -u utils/zetta_submit.py \
  --workspace-name "${workspace_name}" \
  --compute-target "${compute_target}" \
  --experiment-name "${experiment_name}" \
  --display-name "${display_name}" \
  --key-vault-name "exawatt-philly-ipgsp" \
  --docker-address "sramdevregistry.azurecr.io" \
  --docker-name "pytorch:1.13.0-py38-cuda11.6-cudnn8-ubuntu20.04" \
  --local-code-dir "$(pwd)" \
  --cmd "${command}"