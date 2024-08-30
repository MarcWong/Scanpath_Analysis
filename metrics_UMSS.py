# -*- coding: utf-8 -*-
"""
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

def get_gt_files(imgname: str, gt_path: str):
    img_name, task_type = process_image()
    #print(img_name, task_type)
    if imgname not in img_name['filename'].to_list(): return None
    img_id = img_name[img_name['filename'] == imgname]['imageID'].to_numpy()[0]
    return glob(os.path.join(gt_path, f'*fix_{img_id}.tsv'))

def evaluate_UMSS(img_path: str, pred_path:str, gt_path: str):
    list_img_targets = glob(os.path.join(img_path,'*.png'))
    ED = []

    for idx in trange(0, len(list_img_targets)):
        imgfullpath = list_img_targets[idx]
        with Image.open(imgfullpath) as im:
            # width, height = im.size # original image size

            imgname = imgfullpath.split('/')[-1]
            imgname = imgname.strip('.png')
            predCsv = os.path.join(pred_path, f'{imgname}.csv') # all of the predictions are in this file
            if not os.path.exists(predCsv): continue
            df_pred = pd.read_csv(predCsv)

            # GT
            gt_files = get_gt_files(f'{imgname}.png', gt_path)
            if not gt_files: continue

            for gg in range(len(gt_files)):
                df_gt = pd.read_csv(gt_files[gg], sep='\t').dropna()
                if df_gt.empty: continue
                df_gt.columns = ['time', 'index', 'x', 'y']
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
    parser.add_argument('--pred_name', type=str, default='UMSS')
    args = vars(parser.parse_args())

    imgpath = os.path.join('evaluation', 'images')
    gtpath = os.path.join('taskvis', 'fixations')
    predpath = os.path.join('evaluation', 'scanpaths', args['pred_name'])
    evaluate_UMSS(imgpath, predpath, gtpath)