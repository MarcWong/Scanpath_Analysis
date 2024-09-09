# -*- coding: utf-8 -*-
"""
Created on 20240830

@author: Yao Wang
@purpose: to compute the scanpath metrics of predicted scanpaths to the GT scanpaths in the TaskVis dataset
@output : the final scanpath Metrics, LEV↓, DTW↓, SS↑
"""

import os
import json
from glob import glob
import argparse
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from PIL import Image
from tqdm import trange
from pathlib import Path
from utils.metrics import DTW, nw_matching, levenshtein_distance
from utils.utils import get_gt_strings, get_gt_files
from utils.csv_process import process_image

DTWs_A = []
DTWs_B = []
DTWs_C = []
LEVs_A = []
LEVs_B = []
LEVs_C = []
TASK_NAME = ['rv','f','fe']

def calc_metrics(gt_files:list, imgname:str, im:Image, pred_path:str, is_best:bool=False):
    width, height = im.size # original image size
    if 'UMSS' in pred_path:
        predCsv = os.path.join(pred_path, f'{imgname}.csv') # all of the predictions are in this file
        if not os.path.exists(predCsv): return
        df_pred = pd.read_csv(predCsv)
        df_pred.columns = ['user', 'index', 'time', 'x', 'y']
        df_pred = df_pred.drop(['time'],axis=1)
        # Normalization required
        df_pred['x'] = df_pred['x'] * width / 640
        df_pred['y'] = df_pred['y'] * height / 480
    elif 'deepgaze' in pred_path:
        if not os.path.exists(os.path.join(pred_path, f'{imgname}.npy')): return
        predNpy = np.load(os.path.join(pred_path, f'{imgname}.npy'))
        predNpy[:,:,0] = predNpy[:,:,0] * width / 8192
        predNpy[:,:,1] = predNpy[:,:,1] * height / 4096
    elif 'VQA' in pred_path:
        pred = os.path.join(pred_path, 'test_predicts.json')
        metainfo = os.path.join(pred_path, 'AiR_fixations_test.json')
        predJson = json.load(open(pred, 'r'))
        metaJson = json.load(open(metainfo, 'r'))
        img_name, task_type = process_image()
        img_id = img_name[img_name['filename'] == f'{imgname}.png']['imageID'].to_numpy()[0]
        metaJson = [x for x in metaJson if x['image_id'] == f'{img_id}.jpg']

    # GT
    if not gt_files: return
    for t, gt_files_type in enumerate(gt_files):
        if 'ours' in pred_path:
            predCsv = os.path.join(pred_path, f'{imgname}_{TASK_NAME[t]}.csv') # predictions of the task
            if not os.path.exists(predCsv): return
            df_pred = pd.read_csv(predCsv)
            df_pred.columns = ['user', 'index', 'x', 'y']

        DTWs_type = []
        LEVs_type = []
        for gg in range(len(gt_files_type)):
            id_dtw_best = 1000000
            id_lev_best = 1000000
            df_gt = pd.read_csv(gt_files_type[gg], sep='\t').dropna()
            if df_gt.empty: continue
            df_gt.columns = ['time', 'index', 'x', 'y']
            ########## 2D Metrics
            df_gt = df_gt.drop(['index','time'],axis=1).to_numpy()
            ########## 3D Metrics
            #df_gt = df_gt.drop(['index'],axis=1)

            for pp in range(1, len(gt_files_type) + 1):
                if 'UMSS' in pred_path or 'ours' in pred_path:
                    df_predI = df_pred[df_pred['user'] == pp]
                    ########## 2D Metrics
                    df_predI = df_predI.drop(['user', 'index'] ,axis=1).to_numpy()
                elif 'deepgaze' in pred_path:
                    df_predI = predNpy[pp-1]
                elif 'VQA' in pred_path:
                    metaI = metaJson[30*t + pp]
                    predI = [x for x in predJson if x['qid'] == metaI['question_id']]
                    predI = predI[0]
                    df_predI = np.stack((predI['X'], predI['Y']), axis=1)
                if df_predI.size==0: continue
                id_dtw = DTW(df_predI, df_gt)
                id_lev = levenshtein_distance(df_predI, df_gt, width = width, height = height)
                if not is_best:
                    DTWs_type.append(id_dtw)
                    LEVs_type.append(id_lev)
                else:
                    if id_dtw < id_dtw_best:
                        id_dtw_best = id_dtw
                    if id_lev < id_lev_best:
                        id_lev_best = id_lev
            if is_best:
                DTWs_type.append(id_dtw_best)
                LEVs_type.append(id_lev_best)
        if t == 0:
            DTWs_A.extend(DTWs_type)
            LEVs_A.extend(LEVs_type)
        elif t == 1:
            DTWs_B.extend(DTWs_type)
            LEVs_B.extend(LEVs_type)
        elif t == 2:
            DTWs_C.extend(DTWs_type)
            LEVs_C.extend(LEVs_type)

def evaluate_dtw_lev(data_path:str, img_path: str, pred_path:str, is_best:bool):
    """
    @purpose: evaluate the DTW between GT and predicted scanpaths
    @output : the DTW Metrics of A, B, C types
    """
    gt_path = os.path.join(data_path, 'fixations')
    visualisations = glob(os.path.join(img_path,'*.png'))
    for idx in trange(len(visualisations)):
        with Image.open(visualisations[idx]) as im:
            imname = visualisations[idx].split('/')[-1].strip('.png')
            gt_files = get_gt_files(f'{imname}.png', gt_path)
            calc_metrics(gt_files, imname, im, pred_path, is_best)
    return DTWs_A, DTWs_B, DTWs_C, LEVs_A, LEVs_B, LEVs_C

def evaluate_ss(img_path: str, pred_path: str, gt_path: str, is_simplified:bool=False, is_best:bool=False):
    """
    @purpose: evaluate the Sequence Score between GT and predicted scanpaths
    @output : the SS Metrics of A, B, and C types
    """
    SS_A = []
    SS_B = []
    SS_C = []
    strpath = os.path.join(pred_path, 'str')
    visualisations = glob(img_path+'/*.png')
    for idx in trange(len(visualisations)):
        imname = visualisations[idx].split('/')[-1].strip('.png')
        gt_strs = get_gt_strings(gt_path, imname, is_simplified)
        for t, gt_strs_type in enumerate(gt_strs):
            SS_type = []
            for _, string_gt in enumerate(gt_strs_type):
                id_ss_best = 0
                for pp in range(1, len(gt_strs_type) + 1):
                    if 'ours' in pred_path or 'VQA' in pred_path:
                        strfile = f'{strpath}/{pp}/{imname}_{TASK_NAME[t]}.txt'
                    else:
                        strfile = f'{strpath}/{pp}/{imname}.txt'
                    if not os.path.exists(strfile): continue
                    with open(strfile,'r', encoding='utf-8') as f:
                        line = f.readline()
                        id_ss = nw_matching(string_gt, line)
                        if not is_best:
                            SS_type.append(id_ss)
                        elif id_ss > id_ss_best:
                            id_ss_best = id_ss
                if is_best and id_ss_best > 0:
                    SS_type.append(id_ss_best)
            if t == 0:
                SS_A.extend(SS_type)
            elif t == 1:
                SS_B.extend(SS_type)
            elif t == 2:
                SS_C.extend(SS_type)
    return SS_A, SS_B, SS_C

def evaluate_gt(data_path: str, img_path: str, gt_path: str, is_best:bool=False):
    """
    @purpose: evaluate the eigen similarity between the GT scanpaths
    @output : the DTW and SS Metrics of GT scanpaths
    """
    SS_A = []
    SS_B = []
    SS_C = []
    DTWs_A = []
    DTWs_B = []
    DTWs_C = []
    LEVs_A = []
    LEVs_B = []
    LEVs_C = []
    visualisations = glob(os.path.join(img_path,'*.png'))
    for idx in trange(len(visualisations)):
        imname = visualisations[idx].split('/')[-1].strip('.png')
        # if not imname == 'economist_daily_chart_85': continue # only for DEBUG
        with Image.open(visualisations[idx]) as im:
            width, height = im.size
        gt_strs = get_gt_strings(gt_path, imname, False)
        gt_files = get_gt_files(f'{imname}.png', os.path.join(data_path, 'fixations'))
        for t, gt_files_type in enumerate(gt_files):
            DTWs_type = []
            LEVs_type = []
            for gg in range(len(gt_files_type)):
                id_dtw_best = 1000000
                id_lev_best = 1000000
                df_gt = pd.read_csv(gt_files_type[gg], sep='\t').dropna()
                if df_gt.empty: continue
                df_gt.columns = ['time', 'index', 'x', 'y']
                df_gt = df_gt.drop(['index','time'],axis=1).to_numpy()

                for pp in range(len(gt_files_type)):
                    if pp == gg: continue
                    df_gt_2 = pd.read_csv(gt_files_type[pp], sep='\t').dropna()
                    if df_gt_2.empty: continue
                    df_gt_2.columns = ['time', 'index', 'x', 'y']
                    df_gt_2 = df_gt_2.drop(['index','time'],axis=1).to_numpy()
                    id_dtw = DTW(df_gt_2, df_gt)
                    id_lev = levenshtein_distance(df_gt_2, df_gt, width = width, height = height)
                    if not is_best:
                        DTWs_type.append(id_dtw)
                        LEVs_type.append(id_lev)
                    else:
                        if id_dtw < id_dtw_best:
                            id_dtw_best = id_dtw
                        if id_lev < id_lev_best:
                            id_lev_best = id_lev
                if is_best:
                    DTWs_type.append(id_dtw_best)
                    LEVs_type.append(id_lev_best)

            if t == 0:
                DTWs_A.extend(DTWs_type)
                LEVs_A.extend(LEVs_type)
            elif t == 1:
                DTWs_B.extend(DTWs_type)
                LEVs_B.extend(LEVs_type)
            elif t == 2:
                DTWs_C.extend(DTWs_type)
                LEVs_C.extend(LEVs_type)

        for t, gt_strs_type in enumerate(gt_strs):
            SS_type = []
            for gg, string_g in enumerate(gt_strs_type):
                id_ss_best = 0
                for pp, string_p in enumerate(gt_strs_type):
                    if pp == gg: continue
                    id_ss = nw_matching(string_g, string_p)
                    if not is_best:
                        SS_type.append(id_ss)
                    elif id_ss > id_ss_best:
                        id_ss_best = id_ss
                if is_best:
                    SS_type.append(id_ss_best)
            if t == 0:
                SS_A.extend(SS_type)
            elif t == 1:
                SS_B.extend(SS_type)
            elif t == 2:
                SS_C.extend(SS_type)
    return DTWs_A, DTWs_B, DTWs_C, LEVs_A, LEVs_B, LEVs_C, SS_A, SS_B, SS_C

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./taskvis")
    parser.add_argument('--pred_name', type=str, default='UMSS')
    parser.add_argument('--is_simplified_ss', action='store_true')
    parser.add_argument('--best', action='store_true')
    args = vars(parser.parse_args())

    imgpath = os.path.join('evaluation', 'images')
    predpath = os.path.join('evaluation', 'scanpaths', args['pred_name'])
    gtpath = os.path.join('taskvis_analysis', 'fixationsByVis')
    if args['pred_name'] == 'GT':
        DTWs_a, DTWs_b, DTWs_c, LEVs_a, LEVs_b, LEVs_c, SS_A, SS_B, SS_C = evaluate_gt(args['data_path'], imgpath, gtpath, is_best=args['best'])
    else:
        print('evaluating Dynamic Time Warpping')
        DTWs_a, DTWs_b, DTWs_c, LEVs_a, LEVs_b, LEVs_c = evaluate_dtw_lev(args['data_path'],imgpath, predpath, is_best=args['best'])
        print('evaluating Sequence Score')
        SS_A, SS_B, SS_C = evaluate_ss(imgpath, predpath, gtpath, is_simplified=args['is_simplified_ss'], is_best=args['best'])

    print(predpath, np.round(np.mean(SS_A),3), np.round(np.mean(SS_B),3), np.round(np.mean(SS_C),3))
    print(predpath, np.round(np.mean(LEVs_a), 1), np.round(np.mean(LEVs_b), 1), np.round(np.mean(LEVs_c), 1))
    print(predpath, np.round(np.mean(DTWs_a)), np.round(np.mean(DTWs_b)), np.round(np.mean(DTWs_c)))

    df = pd.DataFrame(list(zip(DTWs_a, DTWs_b, DTWs_c, LEVs_a, LEVs_b, LEVs_c, SS_A, SS_B, SS_C)), columns = ['DTW_A', 'DTW_B', 'DTW_C', 'LEV_A', 'LEV_B', 'LEV_C', 'SS_A', 'SS_B', 'SS_C'])
    Path.mkdir(Path('evaluation'), exist_ok=True)
    outname = args["pred_name"]
    if args['best']:
        outname += '_best'
    df.to_csv(f'evaluation/{outname}.csv', index=False)
