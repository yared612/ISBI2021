import cv2
import os, glob
from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from TASK1_dataset import *
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision import models as tm
from TASK1_model import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda'
filename = '/home/yared/文件/ISBI2021/Evaluation_Set/'
# filename = '/home/yared/文件/ISBI2021/Evaluation_gray/'s
file_csv = pd.read_csv('/home/yared/文件/ISBI2021/ensaemble_2_result/WWW_results.csv')
model_path= torch.load('./checkpoint/model_epoch_71.pth')
# model_path= torch.load('./weight/B3/v5/model_epoch_18.pth')
# model_path= torch.load('./weight/v6/model_epoch_79.pth')
m,n = file_csv.shape
image = []
for i in range(0,m):
        name = file_csv.iloc[i, 0]
        image.append(filename + str(name) +'.png')
model = DFModel().cuda()

model.load_state_dict(model_path)
model.eval()
image_    = Image.open(image[0])
mean_transform = transforms.Compose(
        [
            transforms.Resize(512),
            # transforms.CenterCrop(480),
#            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            # transforms.RandomResizedCrop(480),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
        ])
# Y = []
cc = 0
with torch.no_grad():
    for batch_idx, f in enumerate(image):
        name = os.path.basename(f)

        im = mean_transform(Image.open(f)).unsqueeze(0)
#        print(im.type, im.shape)
        
        pred = model(im.to(device))
        pred = torch.sigmoid(pred)
        pred = pred[0].cpu().numpy()
        file_csv.iloc[cc,2:] = pred
        cc = cc+1

#         Y.append([name, pred[1]])
    
# pds = pd.DataFrame(Y)
# pds.columns = ['FileName', 'Glaucoma Risk']
# file_csv.to_csv('WWW_results.csv', index=False)
file_csv.to_csv('./result/B5/WWW_results(71).csv', index=False)

    


