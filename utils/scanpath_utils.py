# -*- coding: utf-8 -*-
"""
Created on 20210603

@author: Yao Wang
@purpose: to compute the scanpath metrics between GT scanpaths in MASSVIS dataset
@output : the final avg. Metrics
"""


import os, sys
from tqdm import trange
from glob import glob
import pandas as pd
import numpy as np

from PIL import Image
pd.options.mode.chained_assignment = None


def pd2string(df:pd.DataFrame) -> str:
    return ''.join(list(map(str, df.values.flatten())))

def remove_duplicates(s: str)->str:
    # remove all non-AOI fixations from the string
    s = s.replace('0', '')
    result = [s[0]]
    for char in s[1:]:
        if char != result[-1]:
            result.append(char)
    return result

def get_gt_files(gt_path:str, imgname:str, extension:str):
    # FIXME: haven't load A/B/C three tytes yet, now DEPRECATED
    gtdir = f'{gt_path}/fixationsByVis/{imgname}/C/'
    if not os.path.exists(gtdir): return '',[]
    for _, _, files in os.walk(gtdir,extension):
        gt_files = files
    return gtdir, gt_files
