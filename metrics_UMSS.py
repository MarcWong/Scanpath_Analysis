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
from utils.metrics import DTW, scaled_TDE

def get_gt_files(imgname: str, gt_path: str) -> list:
    img_name, task_type = process_image()
    # print(img_name, task_type)
    if imgname not in img_name['filename'].to_list(): return None
    img_id = img_name[img_name['filename'] == imgname]['imageID'].to_numpy()[0]
    gt_files_noclass = glob(os.path.join(gt_path, f'*fix_{img_id}.tsv'))
    gt_a = []
    gt_b = []
    gt_c = []
    for gt_file in gt_files_noclass:
        p_name = gt_file.split('/')[-1].split('_')[-3].replace('p','P')
        taskT = task_type.at[p_name,str(img_id)]
        if taskT == 'A':
            gt_a.append(gt_file)
        elif taskT == 'B':
            gt_b.append(gt_file)
        elif taskT == 'C':
            gt_c.append(gt_file)
        else:
            print('Error: Task Type not found')
    return [gt_a, gt_b, gt_c]

def evaluate_UMSS(img_path: str, pred_path:str, gt_path: str):
    list_img_targets = glob(os.path.join(img_path,'*.png'))
    DTWs_a = []
    DTWs_b = []
    DTWs_c = []
    STDEs = []

    for idx in trange(0, len(list_img_targets)):
        imgfullpath = list_img_targets[idx]
        with Image.open(imgfullpath) as im:
            width, height = im.size # original image size

            imgname = imgfullpath.split('/')[-1]
            imgname = imgname.strip('.png')
            predCsv = os.path.join(pred_path, f'{imgname}.csv') # all of the predictions are in this file
            if not os.path.exists(predCsv): continue
            df_pred = pd.read_csv(predCsv)

            # GT
            gt_files = get_gt_files(f'{imgname}.png', gt_path)
            if not gt_files: continue
            for t, gt_file_type in enumerate(gt_files):
                DTWs_type = []

                for gg in range(len(gt_file_type)):
                    df_gt = pd.read_csv(gt_file_type[gg], sep='\t').dropna()
                    if df_gt.empty: continue
                    df_gt.columns = ['time', 'index', 'x', 'y']
                    ########## 2D Metrics
                    df_gt = df_gt.drop(['index','time'],axis=1)
                    ########## 3D Metrics
                    #df_gt = df_gt.drop(['index'],axis=1)

                    for pp in range(1, len(gt_file_type) + 1):
                        df_pred.columns = ['user', 'index', 'time', 'x', 'y']
                        df_predI = df_pred[df_pred['user'] == pp]
                        ########## 2D Metrics
                        df_predI = df_predI.drop(['user', 'index', 'time'] ,axis=1)

                        # Normalization required
                        df_predI['x'] = df_predI['x'] * width / 640
                        df_predI['y'] = df_predI['y'] * height / 480

                        id_dtw = DTW(df_predI, df_gt.to_numpy())
                        # id_stde = scaled_TDE(df_predI,df_gt.to_numpy(), im)

                        DTWs_type.append(id_dtw)
                        # STDEs.append(id_stde)
                if t == 0:
                    DTWs_a.extend(DTWs_type)
                elif t == 1:
                    DTWs_b.extend(DTWs_type)
                elif t == 2:
                    DTWs_c.extend(DTWs_type)

    df = pd.DataFrame(list(zip(DTWs_a,DTWs_b,DTWs_c)), columns = ['DTW_a', 'DTW_b', 'DTW_c'])
    Path.mkdir(Path('evaluation'), exist_ok=True)
    df.to_csv('evaluation/UMSS.csv', index=False)

    print(predpath, np.round(np.mean(DTWs_a), 3), np.round(np.std(DTWs_a), 3))
    print(predpath, np.round(np.mean(DTWs_b), 3), np.round(np.std(DTWs_b), 3))
    print(predpath, np.round(np.mean(DTWs_c), 3), np.round(np.std(DTWs_c), 3))
    # m_STDE = np.asarray(STDEs, dtype=float)
    # print(predpath, np.round(np.mean(m_STDE), 3), np.round(np.std(m_STDE), 3))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_name', type=str, default='UMSS')
    args = vars(parser.parse_args())

    imgpath = os.path.join('evaluation', 'images')
    gtpath = os.path.join('taskvis', 'fixations')
    predpath = os.path.join('evaluation', 'scanpaths', args['pred_name'])
    evaluate_UMSS(imgpath, predpath, gtpath)