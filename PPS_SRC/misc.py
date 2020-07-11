#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tools for finding the lastest folder in postprocessing.
"""
import os
import glob

def find_abs_path_latest_folder(directory):
    """find the lastest folder in the ``directory``, output the latest, absolute path

    Args:

        directory (:obj:`str`): the target directory we will be looking at.

    Returns:

        :obj:`str` : the absolute path of the latest directory in that folder.

    """
    return max(glob.glob(os.path.join(directory, '*/')), key=os.path.getmtime)


def find_relative_path_latest_folder(directory):
    """find the lastest folder in the ``directory``, output the latest, relative path

    Args:

        directory (:obj:`str`): the target directory we will be looking at.

    Returns:

        :obj:`str` : the relative path of the latest directory in that folder.

    """

    abs_path = find_abs_path_latest_folder(directory)
    string_in_slashes = abs_path.split('/')[-2].strip()
    return string_in_slashes
