import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.python.keras import backend as K
from skimage.io import imread,imshow
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
from my_model import *
import cv2

test_filename = '/home/yared/文件/ISBI2021/Evaluation_Set/'
test_file_csv = pd.read_csv('/home/yared/文件/ISBI2021/class_2/result/v1/WWW_results.csv')
test_X = []
m1,n1 = test_file_csv.shape
for tt in range(0,m1):
        name = test_file_csv.iloc[tt, 0]
        test_X.append(test_filename + str(name) +'.png')

imagearray2 = []
# Change the image path with yours.
for path in np.array(test_X):
    img2 = imread(path)
    img2 = cv2.resize(img2, (512, 512))/255
    # met  = img2.mean()
    # stdt = img2.std()
    # nort = (img2 - met) / stdt
    imagearray2.append(img2)
test_x=np.array(imagearray2)

model_name = 'new_model(sigmoid)'
model_path = './saved_models/{}.h5'.format(model_name)
model = ResNet()
model.load_weights(model_path)
y_predict = model.predict(test_x)

for dd in range(0,m1):
    test_file_csv.iloc[dd,2:] = y_predict[dd]
test_file_csv.to_csv('./submissions/v1/WWW_results.csv', index=False)