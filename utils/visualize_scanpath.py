from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from visualization_utils import plot_img, plot_visualization
import json

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