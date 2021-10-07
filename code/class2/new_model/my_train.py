import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.metrics import binary_accuracy
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from skimage.io import imread,imshow
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
from my_model import *
import cv2

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

data_gt = pd.read_csv('/home/yared/文件/ISBI2021/Training_Set/RFMiD_Training_Labels.csv')
filename = '/home/yared/文件/ISBI2021/Training_Set/Training/'
test = glob.glob('/home/yared/文件/ISBI2021/test/*.png')
test_name = []
for tn in test:
    test_n = tn.split('/')[-1].split('.')[0]
    test_name.append(test_n)
test_name.sort()
'''data'''
X,y = [], []
m,n = data_gt.shape
for i in range(0,m):
    name = data_gt.iloc[i, 0]
    if str(name) not in test_name:
        gt_b = data_gt.iloc[i, 2:n]
        gt_array = gt_b.values
        gt_array = gt_array.astype(np.float)
        # gt_a = np.zeros((28,2))
        # for o in range(0,28):
        #     gt_v = gt_array[o]
        #     if gt_v == 1.:
        #         gt_a[o,:] = np.array((0,1))
        #     else:
        #         gt_a[o,:] = np.array((1,0))
        X.append(filename + str(name) +'.png')
        y.append(gt_array)
X_train, X_valid, train_y, valid_y = train_test_split(X,y,test_size=0.2,random_state=0,shuffle=True)
y_train,y_valid = [],[]
for gt in train_y:
    g = gt
    y_train.append(g)
y_train = np.array(y_train)
for gt1 in valid_y:
    g1 = gt1
    y_valid.append(g1)
y_valid = np.array(y_valid)
imagearray = []
imagearray1 = []
for path in np.array(X_train):
    img = imread(path)
    img = cv2.resize(img, (512, 512))/255
    # me  = img.mean()
    # std = img.std()
    # nor = (img - me) / std
    imagearray.append(img)
train_x=np.array(imagearray)
for path in np.array(X_valid):
    img1 = imread(path)
    img1 = cv2.resize(img, (512, 512))/255
    # me1  = img1.mean()
    # std1 = img1.std()
    # nor1 = (img1 - me1) / std1
    imagearray1.append(img1)
valid_x=np.array(imagearray1)
''' data end'''
'''model'''
model_name = 'new_model(sigmoid)'
model = ResNet_sigmoid()
model.summary()
'''model end'''
'''train'''
datagen = ImageDataGenerator(
    rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

optimizer = Adam(lr=10e-4)

model_path = './saved_models/{}.h5'.format(model_name)

checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_binary_accuracy', patience=50, verbose=1)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer, metrics= ['binary_accuracy'])

batch_size = 10
model_history = model.fit_generator(datagen.flow(train_x, y_train, batch_size = batch_size),
                                    epochs = 500,
                                    validation_data = (valid_x, y_valid),
                                    callbacks = [ checkpoint,earlystop])
'''train end'''
'''plot'''
training_loss = model_history.history['loss']
val_loss = model_history.history['val_binary_accuracy']

plt.plot(training_loss, label="training_loss")
plt.plot(val_loss, label="validation_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend(loc='best')
plt.savefig('./loss_fig/' + model_name + '_loss.jpg')
plt.show()

training_acc = model_history.history['binary_accuracy']
val_acc = model_history.history['val_binary_accuracy']

plt.plot(training_acc, label="training_acc")
plt.plot(val_acc, label="validation_acc")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.title("Learning Curve")
plt.legend(loc='best')
plt.savefig('./loss_fig/' + model_name + '_acc.jpg')
plt.show()
'''plot end'''

# model_name = 'new_model'
# model_path = './saved_models/{}.h5'.format(model_name)
# model = ResNet()
# model.load_weights(model_path)
# val_pred = []
# for aa in range (0,346):
#     im = valid_x[aa,:,:,:]
#     im1 = np.reshape(im, (1,512,512,3))
#     y_predict = model.predict(im1)
#     val_pred.append(y_predict)
