#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2020  Microsoft (author: Ke Wang)


def get_premium_storage_by_region_name(region):
    """get associated storage account for a region"""

    region_storage_dict = {
        "eastus":         "exawattaiprmbtts01eus",
        "southcentralus": "exawattaiprmbtts01scus",
        "westus2":        "exawattaiprmbtts01wus2",
        "redmond":        "exawattaiprmbtts01wus2",
        "rrlab":          "exawattaiprmbtts01wus2",
    }

    if region in region_storage_dict:
        return region_storage_dict[region]
    else:
        raise ValueError(
            f"Cannot find the storage account for region {region}")


def get_standard_storage_by_region_name(region):
    """get associated storage account for a region"""

    region_storage_dict = {
        "eastus":         "stdstoragetts01eus",
        "southcentralus": "stdstoragetts01scus",
        "westus2":        "stdstoragetts01wus2",
        "redmond":        "stdstoragetts01wus2",
        "rrlab":          "stdstoragetts01wus2",
    }

    if region in region_storage_dict:
        return region_storage_dict[region]
    else:
        raise ValueError(
            f"Cannot find the storage account for region {region}")


def get_data_storage_by_region_name(region):
    return get_standard_storage_by_region_name(region)


def get_model_storage_by_region_name(region):
    return get_premium_storage_by_region_name(region)


def get_amlt_project_code_storage():
    return "stdstoragetts01eus"
