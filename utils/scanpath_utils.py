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
from natsort import natsorted
pd.options.mode.chained_assignment = None
from metrics import levenshtein_distance, DTW
# from metrics import levenshtein_distance, DTW, scaled_TDE
sys.path.append("..")

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
    # FIXME: haven't load A/B/C three tytes yet
    gtdir = f'{gt_path}/fixationsByVis/{imgname}/C/'
    if not os.path.exists(gtdir): return '',[]
    for _, _, files in os.walk(gtdir,extension):
        gt_files = files
    return gtdir, gt_files

def scanpath_metrics(data_path: str, out_file_path: str):
    imgpath = f'{data_path}/images/'

    list_img_targets = natsorted(glob(imgpath+'*.png'))

    results_DTW, results_LEV, results_REC = [], [], []

    for idx in trange(0, len(list_img_targets)):
        imgfullpath = list_img_targets[idx]
        with Image.open(imgfullpath) as im:
            width, height = im.size # original image size
            # print(f'working on {list_img_targets[idx]}')

            imgname = imgfullpath.replace(imgpath,'')
            imgname = imgname.replace('.png','')

            gtpath, gt_files = get_gt_files(out_file_path, imgname, 'tsv')
            if len(gt_files) == 0: continue

            for pp, _ in enumerate(gt_files):
                df_pred = pd.read_csv(os.path.join(gtpath, gt_files[pp]), header=0, sep='\t')
                df_pred = df_pred.drop(['t'],axis=1)

                DTW_best = 100000
                LEV_best = 100000
                sTDE_best = 0
                for gg, _ in enumerate(gt_files):
                    if pp == gg: continue
                    df_gt = pd.read_csv(os.path.join(gtpath, gt_files[gg]), header=0, sep='\t')
                    df_gt = df_gt.drop(['t'],axis=1)

                    #ed_idx = euclidean_distance(df_pred.to_numpy(),df_gt.to_numpy())
                    tmp_lev = levenshtein_distance(df_pred.to_numpy(), df_gt.to_numpy(), width = width, height = height)
                    #tmp_stde = scaled_TDE(df_pred.to_numpy(),df_gt.to_numpy(), im)
                    tmp_dtw = DTW(df_pred.to_numpy(),df_gt.to_numpy())

                    if tmp_dtw < DTW_best:
                        DTW_best = tmp_dtw
                    if tmp_lev < LEV_best:
                        LEV_best = tmp_lev

                results_DTW.append(DTW_best)
                results_LEV.append(LEV_best)

    results_DTW = np.asarray(results_DTW, dtype=float)
    results_LEV = np.asarray(results_LEV, dtype=float)
    print(np.round(np.mean(results_DTW), 3))
    print(np.round(np.mean(results_LEV), 3))
