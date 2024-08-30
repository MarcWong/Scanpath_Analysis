# -*- coding: utf-8 -*-
""" WIP
Created on 20240830

@author: Yao Wang
@purpose: to compute the scanpath metrics of UMSS predictions to the GT scanpaths in Taskvis dataset
@output : the final avg. Metrics
"""

import pandas as pd
import numpy as np
import os
from glob import glob
import argparse
from PIL import Image
pd.options.mode.chained_assignment = None
from tqdm import trange
from pathlib import Path
from utils.csv_process import process_image
from utils.metrics import DTW, TDE
from get_gt_files import get_gt_files

def get_gt_files(gt_path: str='/netpool/homes/wangyo/Projects/chi2025_scanpath/taskvis/fixations'):
    return os.listdir(gt_path)

def evaluate_UMSS(imgpath: str, predpath:str):
    img_name, task_type = process_image(args['root_path'])
    print(img_name, task_type)
    list_img_targets = glob(imgpath+'*.png')

    ED = []

    for idx in trange(0, len(list_img_targets)):
        imgfullpath = list_img_targets[idx]
        with Image.open(imgfullpath) as im:
            width, height = im.size # original image size
            print('working on %s'% list_img_targets[idx])

            imgname = imgfullpath.replace(imgpath,'')
            imgname = imgname.replace('.png','')
            predCsv = predpath + '%s.csv'%imgname

            if not os.path.exists(predCsv): continue

            df_pred = pd.read_csv(predCsv)

            # 10s / 5s GT
            gtpath, gt_files = get_gt_files(imgname)

            for gg in range(len(gt_files)):
                df_gt = pd.read_csv(os.path.join(gtpath, gt_files[gg]), header=None)
                df_gt.columns = ['index', 'x', 'y', 'time']
                ########## 2D Metrics
                df_gt = df_gt.drop(['index','time'],axis=1)
                ########## 3D Metrics
                #df_gt = df_gt.drop(['index'],axis=1)

                for pp in range(1, len(gt_files) + 1):
                    df_pred.columns = ['user', 'index', 'time', 'x', 'y']
                    df_predI = df_pred[df_pred['user'] == pp]
                    ########## 2D Metrics
                    df_predI = df_predI.drop(['user', 'index', 'time'] ,axis=1)
                    ########## 3D Metrics
                    #df_predI = df_predI.drop(['user', 'index'] ,axis=1)

                    # Normalization required
                    #df_predI['x'] /= 640
                    #df_predI['y'] /= 480
                    #df_predI['x'] *= width
                    #df_predI['y'] *= height

                    #ed_idx = euclidean_distance(df_predI,df_gt)
                    ed_idx = DTW(df_predI, df_gt.to_numpy())
                    #ed_idx = TDE(df_predI,df_gt.to_numpy(), distance_mode='Mean')
                    #ed_idx = scaled_TDE(df_predI,df_gt.to_numpy(), im)

                    ED.append(ed_idx)

    df = pd.DataFrame(list(zip(ED)))
    df.columns = ['DTW']
    Path.mkdir(Path('evaluation'), exist_ok=True)
    df.to_csv('evaluation/UMSS.csv', index=False)

    ED = np.asarray(ED, dtype=float)
    print(predpath, np.round(np.mean(ED), 3))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="/netpool/homes/wangyo/Projects/chi2025_scanpath/evaluation/")
    parser.add_argument('--pred_path', type=str, default='/netpool/homes/wangyo/Projects/chi2025_scanpath/evaluation/scanpaths/UMSS/')
    args = vars(parser.parse_args())

    imgpath = os.path.join(args['root_path'], 'images')
    predpath = os.path.join(args['root_path'], 'scanpaths', 'UMSS')
    evaluate_UMSS(imgpath, predpath)
