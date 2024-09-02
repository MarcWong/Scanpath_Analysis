# -*- coding: utf-8 -*-
"""
@author: Sruthi and Yao
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

def plot_img(im, DPI=100):
    width, height = im.size # original image size
    fig = plt.figure(figsize=(width/DPI,height/DPI))
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.imshow(im)
    return fig

def plot_visualization(x, y, duration):
    plt.xticks([])
    plt.yticks([])
    #colors=plt.cm.rainbow(np.linspace(0,1,40))
    #mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

    plt.plot(x, y, '-o', color='blue',lw=3,alpha=0.4)

    for i in range(len(x)):
        if i == 0:
            li = plt.plot(x[i],y[i],'-o', color='red', ms=25,alpha=0.99,zorder=10)
            plt.setp(li, 'markersize', 25)
        else:
            dur = duration[i] / 10 if duration[i] / 10>10 else 10
            dur = dur if dur<15 else 15
            li = plt.plot(x[i],y[i],'-o', color='yellow', ms=dur,alpha=0.6,zorder=5)
            plt.setp(li, 'markersize', 15)

        #plt.annotate(i, (x[i], y[i]), color='yellow',weight='bold',fontsize=10)

def plot_visualization_colormap(x, y, duration):
    plt.xticks([])
    plt.yticks([])
    colors=plt.cm.rainbow(np.linspace(0,1,40))
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

    plt.plot(x, y, '-ro', color='blue',lw=3,alpha=0.4)
    for i in range(len(x)):
        dur = duration[i] / 10 if duration[i] / 10>10 else 10
        dur = dur if dur<20 else 20
        li = plt.plot(x[i],y[i],'-o',ms=dur,alpha=0.6,zorder=5)
        plt.setp(li, 'markersize', 20)

        #plt.annotate(i, (x[i], y[i]), color='yellow',weight='bold',fontsize=10)

if __name__ == "__main__":
    
    # files = ['3236.png', 'two_col_61188.png', '9120.png', '1379.png', 'two_col_21800.png', '20374873014871.png', 'multi_col_848.png', '35422616009087.png', '4372.png', 'multi_col_20243.png', '14124.png', '08152707004883.png', '42351550020333.png', '9975.png', '2429.png', '56196524000991.png', '13512.png', '933.png', '20767452004312.png', '1270.png', '22651771001645.png', '71354877006926.png', 'two_col_81161.png', 'multi_col_60357.png', 'multi_col_80260.png', 'multi_col_100758.png', '4325.png', '8279.png', 'two_col_1365.png', '5826.png', '10248.png', '3389.png', '91577275004279.png', 'multi_col_703.png', '7503.png', 'multi_col_20328.png', '2721.png', '7880.png', '78055310005226.png', '10265.png']

    image_file = "economist_daily_chart_103.png"

    # data = json.load(open('./data/image_questions.json'))
    # print(data[image_file][question_id])

    image = Image.open('/netpool/homes/wangyo/Dataset/Massvis/targets393/targets/'+image_file)
    fig = plot_img(image)

    df = pd.read_csv('/netpool/homes/wangyo/Dataset/taskvis/fixations/rec_p21_fix_1.tsv', sep='\t')
    # drop last row

    df = df[:-1]
    print(df.tail())
    # print(df.head())
    print(df.head())

    # transform column from strings to integers
    df['RecordingTimestamp'] = df['RecordingTimestamp'].astype(int)

    # get a column values
    x = df['FixationPointX (MCSpx)'].values
    y = df['FixationPointY (MCSpx)'].values
    duration = df['RecordingTimestamp'].values
    print(duration)

    plot_visualization(x,y,duration)

    plt.show()