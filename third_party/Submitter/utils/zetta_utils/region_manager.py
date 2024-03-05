#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2022  Microsoft (author: Ke Wang)

def get_region_by_workspace(workspace_name):
    """get associated region for a workspace"""

    workspace_region_dict = {
        "zetta-prod-ws02-eus2":   "eastus",
        "zetta-amprod-ws01-scus": "southcentralus",
        "zetta-prod-ws01-wus2":   "westus2",
        "zetta-prod-ws03-wus2":   "westus2",
    }

    if workspace_name in workspace_region_dict:
        return workspace_region_dict[workspace_name]
    else:
        raise ValueError(f"Cannot find the region for workspace {workspace_name}")


def get_premium_storage_by_region(region):
    """get associated storage account for a cluster"""

    region_storage_dict = {
        "eastus":         "exawattaiprmbtts01eus",
        "southcentralus": "exawattaiprmbtts01scus",
        "westus2":        "exawattaiprmbtts01wus2",
    }

    if region in region_storage_dict:
        return region_storage_dict[region]
    else:
        raise ValueError(f"Cannot find the premium storage account for {region}")


def get_standard_storage_by_region(region):
    """get associated storage account for a cluster"""

    region_storage_dict = {
        "eastus":         "stdstoragetts01eus",
        "southcentralus": "stdstoragetts01scus",
        "westus2":        "stdstoragetts01wus2",
    }

    if region in region_storage_dict:
        return region_storage_dict[region]
    else:
        raise ValueError(f"Cannot find the standard storage account for {region}")
