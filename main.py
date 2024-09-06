import os, sys
import argparse
from pathlib import Path
from PIL import Image
from tqdm import trange
from glob import glob
import numpy as np
import pandas as pd

from utils.element_utils import get_id_map, get_BBoxes, get_BBoxes_task, plot_element_map
from utils.csv_process import csv_process
from utils.scanpath_utils import pd2string, remove_duplicates

STR = ['Z','a', 'b', 'c', 'T','L','A','M']

def csv_bounding_boxes(data_path: str, out_file_path: str, PLOT_MAP: bool = True):
    """Reads csv files and bounding boxes from disk and write which bounding boxes every fixation belong to.
    :param out_file_path: Where to write the files.
    :param PLOT_MAP: export the plotted maps to disk
    """

    imgpath = f'{data_path}/images/'
    visualisations = glob(imgpath+'*.png')

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
            ques = os.path.basename(question)

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

def scanpath_analysis(data_path: str, out_file_path: str):
    saliency_metrics_list = []

    imgpath = f'{data_path}/images/'
    visualisations = glob(imgpath+'*.png')

    for i in trange(len(visualisations)):
        visualisation = visualisations[i]
        imgname = os.path.basename(visualisation)
        imname, _ = os.path.splitext(imgname)
        all_questions = glob(os.path.join(f'{out_file_path}/fixationsByVis/{imname}' ,'*'))
        for question in all_questions:
            ques = os.path.basename(question)

            participants = glob(os.path.join(question ,'*.tsv'))
            for participant in participants:
                part = os.path.basename(participant)
                p, _ = os.path.splitext(part)
                df = pd.read_csv(participant, index_col = False, header = 0, sep = '\t')
                saccade_len_mean, saccade_len_std = calc_saccade_length(df)

                saliency_metrics = {
                    'participant': p,
                    'image': imname,
                    'question_type': ques,
                    'number of fixations': df.shape[0],
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
    df.to_csv('./scanpath_analysis.tsv', index = False, sep='\t')

# process_str for baseline methods, such as UMSS, deepgaze, etc.
def process_str(data_path: str, img_path: str, pred_path: str):
    visualisations = glob(img_path+'/*.png')

    for idx in trange(len(visualisations)):
        visualisation = visualisations[idx]
        imgname = os.path.basename(visualisation)
        imname, _ = os.path.splitext(imgname)
        bboxes = get_BBoxes(imname, data_path)
        with Image.open(os.path.join(data_path, "images", f"{imname}.png")) as img:
            w, h = img.size
            id_map = get_id_map(bboxes, w, h)

            # UMSS
            if 'UMSS' in pred_path:
                predCsv = os.path.join(pred_path, f'{imname}.csv') # all predictions
                if not os.path.exists(predCsv): continue
                df_pred = pd.read_csv(predCsv)
                df_pred.columns = ['user', 'index', 'time', 'x', 'y']
                for pp in range(1, 47):
                    df_predI = df_pred[df_pred['user'] == pp]
                    scanpath_str = ""
                    # then write the id to the csv
                    for _, row in df_predI.iterrows():
                        scanpath_str += STR[id_map[int(float(row["x"])* w / 640), int(float(row["y"]) * h / 480)]]
                    Path(os.path.join(pred_path, 'str', str(pp))).mkdir(parents=True, exist_ok=True)
                    with open(os.path.join(pred_path, 'str', str(pp), imname+'.txt'), 'w') as f:
                        f.write(str(scanpath_str))
            elif 'deepgaze' in pred_path:
                if not os.path.exists(os.path.join(pred_path, f'{imname}.npy')): continue
                predNpy = np.load(os.path.join(pred_path, f'{imname}.npy')) # all predictions
                predNpy[:,:,0] = predNpy[:,:,0] * w / 8192
                predNpy[:,:,1] = predNpy[:,:,1] * h / 4096
                for pp in range(np.shape(predNpy)[0]):
                    scanpath_str = ""
                    # for fix in range(np.shape(predNpy)[1]):
                    for fix in range(20):
                        scanpath_str += STR[id_map[int(predNpy[pp][fix][0]), int(predNpy[pp][fix][1])]]
                    Path(os.path.join(pred_path, 'str', str(pp))).mkdir(parents=True, exist_ok=True)
                    with open(os.path.join(pred_path, 'str', str(pp), imname+'.txt'), 'w') as f:
                        f.write(str(scanpath_str))
            elif 'ours' in pred_path:
                task_name = ['rv','f','fe']
                for t in range(len(task_name)):
                    predCsv = os.path.join(pred_path, f'{imname}_{task_name[t]}.csv') # predictions of the task
                    if not os.path.exists(predCsv): continue
                    df_pred = pd.read_csv(predCsv)
                    df_pred.columns = ['user', 'index', 'x', 'y']
                    for pp in range(1, 47):
                        df_predI = df_pred[df_pred['user'] == pp]
                        scanpath_str = ""
                        # then write the id to the csv
                        for _, row in df_predI.iterrows():
                            scanpath_str += STR[id_map[int(row["x"]), int(row["y"])]]
                        Path(os.path.join(pred_path, 'str', str(pp))).mkdir(parents=True, exist_ok=True)
                        with open(os.path.join(pred_path, 'str', str(pp), f'{imname}_{task_name[t]}.txt'), 'w') as f:
                            f.write(str(scanpath_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./taskvis")
    parser.add_argument("--out_file_path", type=str, default="./taskvis_analysis")
    parser.add_argument('--process_gt', action='store_true')
    parser.add_argument('--process_baseline', action='store_true')

    args = vars(parser.parse_args())
    Path(args['out_file_path']).mkdir(parents=True, exist_ok=True)

    if args['process_gt']:
        print("processing data...")
        csv_process(args['data_path'], args['out_file_path'])
        print("generating bounding boxes...")
        csv_bounding_boxes(args['data_path'], args['out_file_path'])
        print("analysing scanpaths...")
        scanpath_analysis(args['data_path'], args['out_file_path'])

    if args['process_baseline']:
        print("processing baseline methods...")
        imgpath = os.path.join('evaluation', 'images')
        predpath = os.path.join('evaluation', 'scanpaths', 'UMSS')
        process_str(args['data_path'],imgpath, predpath)
        predpath = os.path.join('evaluation', 'scanpaths', 'deepgaze')
        process_str(args['data_path'],imgpath, predpath)
        predpath = os.path.join('evaluation', 'scanpaths', 'ours')
        process_str(args['data_path'],imgpath, predpath)
