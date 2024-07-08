from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import BBox
from skimage.draw import polygon
from pathlib import Path
import numpy as np


################################################################
## visualization of boxes overlapping on infovis
def bucketColors(idx: int):
    assert idx >= 0
    BUCKETCOLORS = [
        (128/255.,  64/255., 128/255., 0.5),
        (  0/255.,   0/255., 142/255., 0.5),
        (244/255.,  35/255., 232/255., 0.5),
        (102/255., 102/255., 156/255., 0.5),
        (150/255., 120/255.,  90/255., 0.5),
        (250/255., 170/255.,  30/255., 0.5),
        (220/255., 220/255.,        0, 0.5),
        (107/255., 142/255.,  35/255., 0.5),
        ( 70/255.,  70/255.,  70/255., 0.5),
    ]
    return BUCKETCOLORS[idx]

def plot_element_map(impath: str, out_file_path: str, bboxes: list):
    Path(os.path.join(out_file_path, "element_maps_vis")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(out_file_path, "element_maps")).mkdir(parents=True, exist_ok=True)

    with Image.open(impath) as im:
        imgname = os.path.basename(impath)
        imname, ext = os.path.splitext(imgname)

        width, height = im.size # original image size

        # Add the image to the path
        fig = plt.figure(figsize=(width / 72, height / 72), dpi = 72, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        for i,bbox in enumerate(bboxes):
            x = bbox.coords[:,0]
            y = bbox.coords[:,1]
            plt.fill(x, y, color=bucketColors(bbox.id))

        plt.imshow(im)
        plt.axis('off')

        fig.savefig(os.path.join(out_file_path, "element_maps_vis", f"{imname}_element_vis.png"), dpi=fig.dpi)
        plt.close()

        id_map = get_id_map(bboxes, width, height)
        Image.fromarray(np.uint8(id_map), "L").save(os.path.join(out_file_path, "element_maps", f"{imname}_element.png"))


# generate id_map from bboxes
def get_id_map(boxes: list, width: int, height: int)->np.array:
    id_map = np.zeros((width,height), dtype=int)
    # here we set the labels to be 7(mark),6(axis),5(legend),4(title),3(c_task),2(b_task),1(a_task)
    for tt in range(7, 0, -1):
        for box in boxes:
            if box.id == tt:
                rr, cc = polygon(box.coords[:,0], box.coords[:,1], (width, height))
                id_map[rr, cc] = tt

    return id_map


# export bboxes to labelme format csv
def export_labelme(imname: str, out_file_path: str, bboxes: np.ndarray):
    Path(os.path.join(out_file_path, "elementLabels")).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(out_file_path, 'elementLabels', imname), 'w') as export_file:
        for i,_ in enumerate(bboxes):
            #'bboxID','category','x','y'
            for coord in bboxes[i].coords:
                export_file.write(f"{i}, {bboxes[i].id}, {coord[0]}, {coord[1]}\n")

def get_BBoxes(imname: str, data_path: str) -> list:
    annot_path = os.path.join(data_path, "labels_dq", f"{imname}.json")
    with open(annot_path) as json_file:
        annot = json.load(json_file)

    bboxes = []
    if 'shapes' in annot:
        for _, annot_item in enumerate(annot['shapes']):
            if 'label' in annot_item:
                box = BBox.BBox(name = annot_item['label'], \
                        coords = annot_item['points'])
                bboxes.append(box)
    return bboxes

# load task-related bboxes from official labels
def get_BBoxes_task(imname: str, data_path: str) -> list:

    elementLabel = pd.read_csv(os.path.join(data_path, "labels", f"{imname}"), names=['bboxID','category','x','y'])
    elementCoords = []
    elementX = []
    elementY = []
    tmp = 0
    curName = ""

    bboxes = []

    for row in elementLabel.iterrows():
        # row[1][0]: id
        # row[1][1]: category name
        # row[1][2]: x
        # row[1][3]: y

        # a new bbox
        if int(row[1][0]) > tmp:
            # Store the last bbox
            if tmp > 0:
                box = BBox.BBox(name = curName, coords = elementCoords[-1])
                bboxes.append(box)
            tmp = int(row[1][0])
            curName = row[1][1].strip()

            elementCoords.append([])
            elementX.append([])
            elementY.append([])

        elementCoords[-1].append([int(row[1][2]), int(row[1][3])])
        elementX[-1].append(int(row[1][2]))
        elementY[-1].append(int(row[1][3]))

    box = BBox.BBox(name = curName, coords = elementCoords[-1])
    bboxes.append(box)
    return elementCoords, elementX, elementY, bboxes
