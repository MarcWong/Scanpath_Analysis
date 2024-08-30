import os
from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import trange

def process_image(datapath: str):
    """ return two Pandas Frame, imgName and taskType
    :param data_path: Where to read the data
    """
    return pd.read_csv('./taskvis/images.txt', header=0, sep="\t"), \
        pd.read_csv('./taskvis/tasktypes.txt', index_col=0, header=0, sep="\t")

def csv_process(data_path: str, out_file_path: str):
    """ Reads csv files and saves them separately in folders:
    :param data_path: Where to read the data
    :param out_file_path: Where to write the files
    """
    img_name, task_type = process_image(data_path)

    for i in trange(1,31):
        scanpaths_path = glob(os.path.join(data_path, 'fixations' ,f'*_{i}.tsv'))
        for scanpath_path in scanpaths_path:
            p_name = scanpath_path.split('_')[-3].replace('p','P')
            scanpath = pd.read_csv(scanpath_path, header=0, sep="\t")
            scanpath.drop([len(scanpath)-1], inplace= True)
            if not scanpath.empty and not img_name[img_name['imageID'] == i].empty:
                image_name = img_name[img_name['imageID'] == i]['filename'].to_numpy()[0]
                taskT = task_type.at[p_name,str(i)]
                Path(os.path.join(out_file_path, "fixationsByVis")).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(out_file_path, "fixationsByVis", image_name.strip('.png'))).mkdir(parents=True, exist_ok=True)
                Path(os.path.join(out_file_path, "fixationsByVis", image_name.strip('.png'), taskT)).mkdir(parents=True, exist_ok=True)
                #scanpath['duration'] = scanpath['RecordingTimestamp'].diff()
                #scanpath.at[0, 'duration'] = scanpath['gRecordingTimestamp'].iloc[0]
                scanpath.dropna(axis=0, how='any', inplace=True)
                scanpath['RecordingTimestamp'] = (scanpath['FixationIndex'] -1) * 200
                scanpath = scanpath.drop(['FixationIndex'], axis=1)
                scanpath.columns=['t','x','y']
                scanpath['t'] = scanpath['t'].astype(int)
                scanpath['x'] = scanpath['x'].astype(int)
                scanpath['y'] = scanpath['y'].astype(int)
                scanpath.to_csv(os.path.join(out_file_path, "fixationsByVis", image_name.strip('.png'), taskT, p_name+'.tsv'), index = False, sep='\t')
