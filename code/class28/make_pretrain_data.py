import pandas as pd
from skimage.io import imread,imsave
import numpy as np
import glob
filename = '/home/yared/文件/ISBI2021/Training_Set/Training/'
data_gt = pd.read_csv('/home/yared/文件/ISBI2021/Training_Set/RFMiD_Training_Labels.csv')
save_path = '/home/yared/文件/ISBI2021/class_2/ju/img/'
m,n = data_gt.shape

test = glob.glob('/home/yared/文件/ISBI2021/test/*.png')
test_name = []
for tn in test:
    test_n = tn.split('/')[-1].split('.')[0]
    test_name.append(test_n)
test_name.sort()

cnt = 0
for i in range(0,m):
    name = data_gt.iloc[i, 0]
    gt = data_gt.iloc[i, 1]
    if str(name)  in test_name:
        if gt == 1:
            cnt = cnt + 1
        # img = imread(filename + str(name) +'.png')
        # if gt == 1:
        #     imsave(save_path + 'bad/'  + str(name) +'.png',img)
        # else:
        #     imsave(save_path + 'good/'  + str(name) +'.png',img)
        #     imsave(save_path + 'good/'  + str(name) +'_1.png',img)