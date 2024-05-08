#!/usr/bin/env python -u
# -*- coding: utf-8 -*-

# Copyright  2020  Microsoft (author: Ke Wang)

from __future__ import absolute_import, division, print_function


def str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ['true', 'false']:
        raise ValueError('Argument needs to be a boolean, got {}'.format(s))
    return {'true': True, 'false': False}[s.lower()]
