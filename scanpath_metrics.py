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
from utils.metrics import DTW, scaled_TDE, nw_matching
from tqdm import trange
from utils.utils import get_gt_strings, get_gt_files
    

def evaluate_deepgaze(data_path:str, img_path: str, pred_path:str):
    gt_path = os.path.join(data_path, 'fixations')
    visualisations = glob(os.path.join(img_path,'*.png'))
    DTWs_a = []
    DTWs_b = []
    DTWs_c = []
    STDEs_a = []
    STDEs_b = []
    STDEs_c = []
    for idx in trange(len(visualisations)):
        imgfullpath = visualisations[idx]
        with Image.open(imgfullpath) as im:
            width, height = im.size # original image size
            imgname = imgfullpath.split('/')[-1]
            imgname = imgname.strip('.png')
            predNpy = np.load(os.path.join(pred_path, f'{imgname}.npy'))
            predNpy[:,:,0] = predNpy[:,:,0] * width / 8192
            predNpy[:,:,1] = predNpy[:,:,1] * height / 4096

            # GT
            gt_files = get_gt_files(f'{imgname}.png', gt_path)
            if not gt_files: continue
            for t, gt_files_type in enumerate(gt_files):
                DTWs_type = []
                STDEs_type = []

                for gg in range(len(gt_files_type)):
                    df_gt = pd.read_csv(gt_files_type[gg], sep='\t').dropna()
                    if df_gt.empty: continue
                    df_gt.columns = ['time', 'index', 'x', 'y']
                    ########## 2D Metrics
                    df_gt = df_gt.drop(['index','time'],axis=1)
                    ########## 3D Metrics
                    #df_gt = df_gt.drop(['index'],axis=1)

                    for pp in range(1, len(gt_files_type) + 1):
                        id_dtw = DTW(predNpy[pp-1], df_gt.to_numpy())
                        id_stde = scaled_TDE(predNpy[pp-1],df_gt.to_numpy(), im)

                        DTWs_type.append(id_dtw)
                        STDEs_type.append(id_stde)
                if t == 0:
                    DTWs_a.extend(DTWs_type)
                    STDEs_a.extend(STDEs_type)
                elif t == 1:
                    DTWs_b.extend(DTWs_type)
                    STDEs_b.extend(STDEs_type)
                elif t == 2:
                    DTWs_c.extend(DTWs_type)
                    STDEs_c.extend(STDEs_type)

    df = pd.DataFrame(list(zip(DTWs_a,DTWs_b,DTWs_c, STDEs_a, STDEs_b, STDEs_c)), columns = ['DTW_a', 'DTW_b', 'DTW_c', 'STDE_a', 'STDE_b', 'STDE_c'])
    Path.mkdir(Path('evaluation'), exist_ok=True)
    df.to_csv('evaluation/deepgaze.csv', index=False)

    print(predpath, np.round(np.mean(STDEs_a), 3), np.round(np.std(STDEs_a), 3))
    print(predpath, np.round(np.mean(STDEs_b), 3), np.round(np.std(STDEs_b), 3))
    print(predpath, np.round(np.mean(STDEs_c), 3), np.round(np.std(STDEs_c), 3))

def evaluate_UMSS(data_path:str, img_path: str, pred_path:str):
    gt_path = os.path.join(data_path, 'fixations')
    visualisations = glob(os.path.join(img_path,'*.png'))
    DTWs_a = []
    DTWs_b = []
    DTWs_c = []
    STDEs_a = []
    STDEs_b = []
    STDEs_c = []
    for idx in trange(len(visualisations)):
        imgfullpath = visualisations[idx]
        with Image.open(imgfullpath) as im:
            width, height = im.size # original image size
            imgname = imgfullpath.split('/')[-1]
            imgname = imgname.strip('.png')
            predCsv = os.path.join(pred_path, f'{imgname}.csv') # all of the predictions are in this file
            if not os.path.exists(predCsv): continue
            df_pred = pd.read_csv(predCsv)
            df_pred.columns = ['user', 'index', 'time', 'x', 'y']

            # GT
            gt_files = get_gt_files(f'{imgname}.png', gt_path)
            if not gt_files: continue
            for t, gt_files_type in enumerate(gt_files):
                DTWs_type = []
                STDEs_type = []

                for gg in range(len(gt_files_type)):
                    df_gt = pd.read_csv(gt_files_type[gg], sep='\t').dropna()
                    if df_gt.empty: continue
                    df_gt.columns = ['time', 'index', 'x', 'y']
                    ########## 2D Metrics
                    df_gt = df_gt.drop(['index','time'],axis=1)
                    ########## 3D Metrics
                    #df_gt = df_gt.drop(['index'],axis=1)

                    for pp in range(1, len(gt_files_type) + 1):
                        df_predI = df_pred[df_pred['user'] == pp]
                        ########## 2D Metrics
                        df_predI = df_predI.drop(['user', 'index', 'time'] ,axis=1)

                        # Normalization required
                        df_predI['x'] = df_predI['x'] * width / 640
                        df_predI['y'] = df_predI['y'] * height / 480

                        id_dtw = DTW(df_predI, df_gt.to_numpy())
                        id_stde = scaled_TDE(df_predI,df_gt.to_numpy(), im)

                        DTWs_type.append(id_dtw)
                        STDEs_type.append(id_stde)
                if t == 0:
                    DTWs_a.extend(DTWs_type)
                    STDEs_a.extend(STDEs_type)
                elif t == 1:
                    DTWs_b.extend(DTWs_type)
                    STDEs_b.extend(STDEs_type)
                elif t == 2:
                    DTWs_c.extend(DTWs_type)
                    STDEs_c.extend(STDEs_type)

    df = pd.DataFrame(list(zip(DTWs_a,DTWs_b,DTWs_c, STDEs_a, STDEs_b, STDEs_c)), columns = ['DTW_a', 'DTW_b', 'DTW_c', 'STDE_a', 'STDE_b', 'STDE_c'])
    Path.mkdir(Path('evaluation'), exist_ok=True)
    df.to_csv('evaluation/UMSS.csv', index=False)

    print(predpath, np.round(np.mean(STDEs_a), 3), np.round(np.std(STDEs_a), 3))
    print(predpath, np.round(np.mean(STDEs_b), 3), np.round(np.std(STDEs_b), 3))
    print(predpath, np.round(np.mean(STDEs_c), 3), np.round(np.std(STDEs_c), 3))

def evaluate_SS(imgpath: str, strpath: str, gtpath: str, is_simplified=False):
    visualisations = glob(imgpath+'/*.png')
    SS_A = []
    SS_B = []
    SS_C = []
    for idx in trange(len(visualisations)):
        imgfullpath = visualisations[idx]
        imgname = imgfullpath.split('/')[-1]
        imname = imgname.strip('.png')
        gt_strs = get_gt_strings(gtpath, imname, is_simplified)
        for t, gt_strs_type in enumerate(gt_strs):
            SS_type = []
            for gg in range(len(gt_strs_type)):
                strings = gt_strs_type[gg]
                for pp in range(1, len(gt_strs_type) + 1):
                    strfile = f'{strpath}/{pp}/{imname}.txt'
                    f = open(strfile,'r')
                    line = f.readline()
                    res = nw_matching(strings, line)
                    SS_type.append(res)
            if t == 0:
                SS_A.extend(SS_type)
            elif t == 1:
                SS_B.extend(SS_type)
            elif t == 2:
                SS_C.extend(SS_type)

    df = pd.DataFrame(list(zip(SS_A, SS_B, SS_C)))
    df.columns = ['SS_A', 'SS_B', 'SS_C']
    df.to_csv(f'evaluation/{strpath.split("/")[-2]}_ss.csv', index=False)
    print(np.mean(np.array(SS_A)), np.mean(np.array(SS_B)), np.mean(np.array(SS_C)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./taskvis")
    parser.add_argument('--pred_name', type=str, default='UMSS')
    parser.add_argument('--is_simplified_ss', action='store_true')
    args = vars(parser.parse_args())

    imgpath = os.path.join('evaluation', 'images')
    predpath = os.path.join('evaluation', 'scanpaths', args['pred_name'])
    if args['pred_name'] == 'UMSS':
        evaluate_UMSS(args['data_path'],imgpath, predpath)
    elif args['pred_name'] == 'deepgaze':
        evaluate_deepgaze(args['data_path'],imgpath, predpath)

    gtpath = os.path.join('taskvis_analysis', 'fixationsByVis')
    strpath = os.path.join('evaluation', 'scanpaths', args['pred_name'], 'str')
    evaluate_SS(imgpath, strpath, gtpath, is_simplified=args['is_simplified_ss'])
