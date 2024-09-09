import os, sys
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import trange
from glob import glob
import numpy as np
import pandas as pd

from utils.element_utils import get_id_map, get_BBoxes, get_BBoxes_task, plot_element_map
from utils.csv_process import csv_process, process_image
from utils.scanpath_utils import pd2string, remove_duplicates

STR = ['Z','a', 'b', 'c', 'T','L','A','M']

def csv_bounding_boxes(data_path: str, out_file_path: str, PLOT_MAP: bool = True):
    """Reads csv files and bounding boxes from disk and write which bounding boxes every fixation belong to.

    Args:
        data_path (str): data path to read the files.
        out_file_path (str): Where to write the files.
        PLOT_MAP (bool, optional): export the plotted maps to disk. Defaults to True.
    """
    img_path = f'{data_path}/images/'
    visualisations = glob(img_path+'*.png')

    for i in trange(len(visualisations)):
        visualisation = visualisations[i]
        imgname = os.path.basename(visualisation)
        imname, ext = os.path.splitext(imgname)

        bboxes = get_BBoxes(imname, data_path)
        _,_,_,task_bboxes = get_BBoxes_task(imname, data_path)

        # Plot element maps to a folder
        if PLOT_MAP: plot_element_map(os.path.join(data_path, "images", f"{imname}.png"), out_file_path, task_bboxes)

        with Image.open(os.path.join(data_path, "images", f"{imname}.png")) as img:
            w, h = img.size
            id_map = get_id_map(bboxes, w, h)
            id_map_task = get_id_map(task_bboxes, w, h)

        # writing participant csvs
        all_questions = glob(os.path.join(f'{out_file_path}/fixationsByVis/{imname}' ,'*'))
        for question in all_questions:
            participants = glob(os.path.join(question ,'*.tsv'))
            for participant in participants:
                scanpath_str = ""
                new_list = []
                df = pd.read_csv(participant, index_col = False, header = 0, sep = '\t')

                # then write the id to the csv
                for _, row in df.iterrows():
                    new_list.append([row["x"], row["y"], row["t"], \
                        id_map[int(row["x"]), int(row["y"])], \
                        STR[id_map[int(row["x"]), int(row["y"])]], \
                        id_map_task[int(row["x"]), int(row["y"])]])
                    scanpath_str += STR[id_map[int(row["x"]), int(row["y"])]]

                df_new = pd.DataFrame(new_list, columns = ['x', 'y', 't', 'id', 'label', 'task_id'])
                df_new.to_csv(participant, index = False, sep='\t')

                with open(os.path.join(question, f'{participant.split("/")[-1].strip(".tsv")}.txt'),'w') as f:
                    f.write(str(scanpath_str));

def calc_aoi_shift(df: pd.DataFrame) -> float:
    df_str = pd2string(df)
    df_str = remove_duplicates(df_str)
    return len(df_str)

def calc_revisit_freq(df: pd.DataFrame, id: int) -> float:
    df_str = pd2string(df)
    df_str = remove_duplicates(df_str)
    return df_str.count(str(id))

def calc_saccade_length(df: pd.DataFrame) -> (float, float):
    if not df['x'].empty:
        x = df['x'].to_numpy()
        y = df['y'].to_numpy()
        saccade_length = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        return np.round(np.mean(saccade_length),3), np.round(np.std(saccade_length),3)
    return 0, 0

def calc_ftot(df: pd.DataFrame, ques: str) -> float:
    df_str = pd2string(df)
    if ques == 'A':
        return (df_str.find('1'))
    if ques == 'B':
        return (df_str.find('2'))
    if ques == 'C':
        return (df_str.find('3'))
    return -1

def calc_task_ratio(df: pd.DataFrame, ques: str) -> float:
    df_str = pd2string(df)
    if ques == 'A':
        return np.round(df_str.count('1') / len(df_str), 4)
    if ques == 'B':
        return np.round(df_str.count('2') / len(df_str), 4)
    if ques == 'C':
        return np.round(df_str.count('3') / len(df_str), 4)
    return -1

def print_m_std(df: pd.DataFrame, col: str, is_ratio: bool = False):
    if is_ratio:
        print(f'{col}_mean', np.round(np.mean(df[col]) * 100, 1))
        print(f'{col}_std', np.round(np.std(df[col]) * 100, 1))
    else:
        print(f'{col}_mean', np.round(np.mean(df[col]), 1))
        print(f'{col}_std', np.round(np.std(df[col]), 1))

def print_df(df: pd.DataFrame):
    print_m_std(df, 'number_of_fixations')
    print_m_std(df, 'saccade_len_m')
    print_m_std(df, 'fixation_task_ratio', is_ratio=True)
    print_m_std(df, 'title_ratio', is_ratio=True)
    print_m_std(df, 'mark_ratio', is_ratio=True)
    print_m_std(df, 'axis_ratio', is_ratio=True)
    print_m_std(df, 'aoi_shift')
    print_m_std(df, 'revisit_freq_title')
    print_m_std(df, 'revisit_freq_mark')
    print_m_std(df, 'revisit_freq_axis')

def print_metrics(df: pd.DataFrame, separate_type: bool = False):
    question_types = ['A', 'B', 'C']
    if separate_type:
        for qtype in question_types:
            dft= df[df['question_type'] == qtype]
            print(f'Question type: {qtype}')
            print_df(dft)
    else:
        print_df(df)

def scanpath_analysis(img_path: str, gt_path: str, out_path: str, is_eval: bool = False):
    saliency_metrics_list = []
    visualisations = glob(img_path+'/*.png')

    for i in trange(len(visualisations)):
        visualisation = visualisations[i]
        imgname = os.path.basename(visualisation)
        imname, _ = os.path.splitext(imgname)
        all_questions = glob(os.path.join(f'{gt_path}/fixationsByVis/{imname}', '*'))
        for question in all_questions:
            ques = os.path.basename(question)

            participants = glob(os.path.join(question, '*.tsv'))
            for participant in participants:
                part = os.path.basename(participant)
                p, _ = os.path.splitext(part)
                df = pd.read_csv(participant, index_col = False, header = 0, sep = '\t')
                saccade_len_mean, saccade_len_std = calc_saccade_length(df)

                saliency_metrics = {
                    'participant': p,
                    'image': imname,
                    'question_type': ques,
                    'number_of_fixations': df.shape[0],
                    'saccade_len_m': saccade_len_mean,
                    'saccade_len_std': saccade_len_std,
                    'aoi_shift': calc_aoi_shift(df['id']),
                    'first_time_on_task': calc_ftot(df['task_id'], ques),
                    'fixation_task_ratio': calc_task_ratio(df['task_id'], ques),
                    'title_ratio': np.round(df[df['id']==4].shape[0] / df.shape[0],4),
                    'legend_ratio': np.round(df[df['id']==5].shape[0] / df.shape[0], 4),
                    'axis_ratio': np.round(df[df['id']==6].shape[0] / df.shape[0], 4),
                    'mark_ratio': np.round(df[df['id']==7].shape[0] / df.shape[0], 4),
                    'revisit_freq_title': calc_revisit_freq(df['id'], 4),
                    'revisit_freq_legend': calc_revisit_freq(df['id'], 5),
                    'revisit_freq_axis': calc_revisit_freq(df['id'], 6),
                    'revisit_freq_mark': calc_revisit_freq(df['id'], 7),
                    'scanpath_str': pd2string(df['label'])
                }
                saliency_metrics_list.append(saliency_metrics)
    df = pd.DataFrame.from_dict(saliency_metrics_list)
    print_metrics(df, separate_type=True)
    df.to_csv(out_path, index = False, sep='\t')

def scanpath_analysis_eval(pred_path: str, output_path: str, separate_type: bool = False):
    saliency_metrics_list = []
    questions = ['A', 'B', 'C']
    task_names = ['rv', 'f', 'fe']

    all_participants = glob(os.path.join(pred_path, 'str', '*'))
    for participant in all_participants:
        part = os.path.basename(participant)
        p, _ = os.path.splitext(part)
        for i, ques in enumerate(questions):
            if 'ours' in pred_path or 'VQA' in pred_path:
                img_paths = glob(os.path.join(participant, f'*{task_names[i]}.tsv'))
            else:
                img_paths = glob(os.path.join(participant, '*.tsv'))
            for imgpath in img_paths:
                imgname = os.path.basename(imgpath)
                imname, _ = os.path.splitext(imgname)
                df = pd.read_csv(imgpath, index_col = False, header = 0, sep = '\t')
                if len(df) == 0: continue
                saccade_len_mean, saccade_len_std = calc_saccade_length(df)

                saliency_metrics = {
                    'participant': p,
                    'image': imname,
                    'question_type': ques,
                    'number_of_fixations': df.shape[0],
                    'saccade_len_m': saccade_len_mean,
                    'saccade_len_std': saccade_len_std,
                    'aoi_shift': calc_aoi_shift(df['id']),
                    'first_time_on_task': calc_ftot(df['task_id'], ques),
                    'fixation_task_ratio': calc_task_ratio(df['task_id'], ques),
                    'title_ratio': np.round(df[df['id']==4].shape[0] / df.shape[0],4),
                    'legend_ratio': np.round(df[df['id']==5].shape[0] / df.shape[0], 4),
                    'axis_ratio': np.round(df[df['id']==6].shape[0] / df.shape[0], 4),
                    'mark_ratio': np.round(df[df['id']==7].shape[0] / df.shape[0], 4),
                    'revisit_freq_title': calc_revisit_freq(df['id'], 4),
                    'revisit_freq_legend': calc_revisit_freq(df['id'], 5),
                    'revisit_freq_axis': calc_revisit_freq(df['id'], 6),
                    'revisit_freq_mark': calc_revisit_freq(df['id'], 7),
                    'scanpath_str': pd2string(df['label'])
                }
                saliency_metrics_list.append(saliency_metrics)
    df = pd.DataFrame.from_dict(saliency_metrics_list)
    print_metrics(df, separate_type=separate_type)

    df.to_csv(output_path, index = False, sep='\t')

def process_str(data_path: str, img_path: str, pred_path: str):
    """process_str for baseline methods, such as UMSS, deepgaze, etc.

    Args:
        data_path (str): _description_
        img_path (str): _description_
        pred_path (str): _description_
    """    
    visualisations = glob(img_path+'/*.png')

    for idx in trange(len(visualisations)):
        visualisation = visualisations[idx]
        imgname = os.path.basename(visualisation)
        imname, _ = os.path.splitext(imgname)
        bboxes = get_BBoxes(imname, data_path)
        _,_,_,task_bboxes = get_BBoxes_task(imname, data_path)
        with Image.open(os.path.join(data_path, "images", f"{imname}.png")) as img:
            w, h = img.size
            id_map = get_id_map(bboxes, w, h)
            id_map_task = get_id_map(task_bboxes, w, h)
            task_name = ['rv','f','fe']

            if 'UMSS' in pred_path:
                predCsv = os.path.join(pred_path, f'{imname}.csv') # all predictions
                if not os.path.exists(predCsv): continue
                df_pred = pd.read_csv(predCsv)
                df_pred.columns = ['user', 'index', 'time', 'x', 'y']
                for pp in range(1, 31):
                    new_list = []
                    df_predI = df_pred[df_pred['user'] == pp]
                    scanpath_str = ""
                    # then write the id to the csv
                    for _, row in df_predI.iterrows():
                        xx = int(float(row["x"])* w / 640)
                        yy = int(float(row["y"]) * h / 480)
                        scanpath_str += STR[id_map[int(xx), int(yy)]]
                        new_list.append([xx, yy, id_map[xx, yy],STR[id_map[xx, yy]], id_map_task[xx, yy]])
                    Path(os.path.join(pred_path, 'str', str(pp))).mkdir(parents=True, exist_ok=True)
                    with open(os.path.join(pred_path, 'str', str(pp), imname+'.txt'), 'w') as f:
                        f.write(str(scanpath_str))
                    df_new = pd.DataFrame(new_list, columns = ['x', 'y', 'id', 'label', 'task_id'])
                    df_new.to_csv(os.path.join(pred_path, 'str', str(pp), imname+'.tsv'), index = False, sep='\t')
            elif 'deepgaze' in pred_path:
                if not os.path.exists(os.path.join(pred_path, f'{imname}.npy')): continue
                predNpy = np.load(os.path.join(pred_path, f'{imname}.npy')) # all predictions
                predNpy[:,:,0] = predNpy[:,:,0] * w / 8192
                predNpy[:,:,1] = predNpy[:,:,1] * h / 4096
                for pp in range(np.shape(predNpy)[0]):
                    new_list = []
                    scanpath_str = ""
                    for fix in range(np.shape(predNpy)[1]):
                        xx = int(predNpy[pp][fix][0])
                        yy = int(predNpy[pp][fix][1])
                        scanpath_str += STR[id_map[xx, yy]]
                        new_list.append([xx, yy, id_map[xx, yy],STR[id_map[xx, yy]], id_map_task[xx, yy]])
                    Path(os.path.join(pred_path, 'str', str(pp))).mkdir(parents=True, exist_ok=True)
                    with open(os.path.join(pred_path, 'str', str(pp), imname+'.txt'), 'w') as f:
                        f.write(str(scanpath_str))
                    df_new = pd.DataFrame(new_list, columns = ['x', 'y', 'id', 'label', 'task_id'])
                    df_new.to_csv(os.path.join(pred_path, 'str', str(pp), imname+'.tsv'), index = False, sep='\t')
            elif 'ours' in pred_path:
                for t in range(len(task_name)):
                    predCsv = os.path.join(pred_path, f'{imname}_{task_name[t]}.csv') # predictions of the task
                    if not os.path.exists(predCsv): continue
                    df_pred = pd.read_csv(predCsv)
                    df_pred.columns = ['user', 'index', 'x', 'y']
                    for pp in range(1, 31):
                        df_predI = df_pred[df_pred['user'] == pp]
                        new_list = []
                        scanpath_str = ""
                        for _, row in df_predI.iterrows():
                            xx = int(row["x"])
                            yy = int(row["y"])
                            scanpath_str += STR[id_map[xx, yy]]
                            new_list.append([xx, yy, id_map[xx, yy],STR[id_map[xx, yy]], id_map_task[xx, yy]])
                        Path(os.path.join(pred_path, 'str', str(pp))).mkdir(parents=True, exist_ok=True)
                        with open(os.path.join(pred_path, 'str', str(pp), f'{imname}_{task_name[t]}.txt'), 'w') as f:
                            f.write(str(scanpath_str))
                        df_new = pd.DataFrame(new_list, columns = ['x', 'y', 'id', 'label', 'task_id'])
                        df_new.to_csv(os.path.join(pred_path, 'str', str(pp), f'{imname}_{task_name[t]}.tsv'), index = False, sep='\t')
            elif 'VQA' in pred_path:
                pred = os.path.join(pred_path, 'test_predicts.json')
                metainfo = os.path.join(pred_path, 'AiR_fixations_test.json')
                predJson = json.load(open(pred, 'r'))
                metaJson = json.load(open(metainfo, 'r'))
                img_name, task_type = process_image()
                img_id = img_name[img_name['filename'] == imgname]['imageID'].to_numpy()[0]
                metaJson = [x for x in metaJson if x['image_id'] == f'{img_id}.jpg']
                for t in range(len(task_name)):
                    for pp in range(30):
                        new_list = []
                        scanpath_str = ""
                        metaI = metaJson[30*t + pp]
                        predI = [x for x in predJson if x['qid'] == metaI['question_id']]
                        predI = predI[0]
                        df_predI = np.stack((predI['X'], predI['Y']), axis=1)
                        for fix in range(df_predI.shape[0]):
                            xx = int(df_predI[fix][0])
                            yy = int(df_predI[fix][1])
                            scanpath_str += STR[id_map[xx, yy]]
                            new_list.append([xx, yy, id_map[xx, yy],STR[id_map[xx, yy]], id_map_task[xx, yy]])
                        Path(os.path.join(pred_path, 'str', str(pp))).mkdir(parents=True, exist_ok=True)
                        with open(os.path.join(pred_path, 'str', str(pp), f'{imname}_{task_name[t]}.txt'), 'w') as f:
                            f.write(str(scanpath_str))
                        df_new = pd.DataFrame(new_list, columns = ['x', 'y', 'id', 'label', 'task_id'])
                        df_new.to_csv(os.path.join(pred_path, 'str', str(pp), f'{imname}_{task_name[t]}.tsv'), index = False, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./taskvis")
    parser.add_argument("--out_file_path", type=str, default="./taskvis_analysis")
    parser.add_argument('--process_gt', action='store_true')
    parser.add_argument('--process_baseline', action='store_true')
    parser.add_argument('--analysis_baseline', action='store_true')
    args = vars(parser.parse_args())
    Path(args['out_file_path']).mkdir(parents=True, exist_ok=True)

    if args['process_gt']:
        print("processing data...")
        csv_process(args['data_path'], args['out_file_path'])
        print("generating bounding boxes...")
        csv_bounding_boxes(args['data_path'], args['out_file_path'])
        print("analysing scanpaths...")
        scanpath_analysis(os.path.join(args['data_path'], 'images'), args['out_file_path'], 'scanpath_analysis.tsv')

    if args['process_baseline']:
        print("processing baseline methods...")
        imgpath = os.path.join('evaluation', 'images')
        process_str(args['data_path'],imgpath, os.path.join('evaluation', 'scanpaths', 'deepgaze'))
        process_str(args['data_path'],imgpath, os.path.join('evaluation', 'scanpaths', 'VQA'))
        process_str(args['data_path'],imgpath, os.path.join('evaluation', 'scanpaths', 'UMSS'))
        process_str(args['data_path'],imgpath, os.path.join('evaluation', 'scanpaths', 'ours'))

    if args['analysis_baseline']:
        print("analysing Human...")
        scanpath_analysis(os.path.join('evaluation', 'images'), args['out_file_path'], os.path.join('evaluation', 'human_eval.tsv'), is_eval=True)
        print("analysing deepgaze...")
        scanpath_analysis_eval(os.path.join('evaluation', 'scanpaths', 'deepgaze'), os.path.join('evaluation', 'deepgaze_analysis.tsv'))
        print("analysing VQA...")
        scanpath_analysis_eval(os.path.join('evaluation', 'scanpaths', 'VQA'), os.path.join('evaluation', 'VQA_analysis.tsv'), separate_type=True)
        print("analysing UMSS...")
        scanpath_analysis_eval(os.path.join('evaluation', 'scanpaths', 'UMSS'), os.path.join('evaluation', 'UMSS_analysis.tsv'))
        print("analysing ours...")
        scanpath_analysis_eval(os.path.join('evaluation', 'scanpaths', 'ours'), os.path.join('evaluation', 'ours_analysis.tsv'), separate_type=True)
