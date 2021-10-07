# import cv2
import os, glob
from torchvision import models
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torchvision import models as tm
# from torchsummary import summary
from TASK1_model import *
from TASK1_dataset import *
from sklearn.model_selection import train_test_split
import pandas as pd
import glob
from Dice_loss import*
from my_losses import *
from ranger import *
from asam import *
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # torch.cuda.set_device(0)
    # model_path= torch.load('./checkpoint_MCC/model_epoch_199.pth')
    test = glob.glob('/home/yared/文件/ISBI2021/test/*.png')
    test_name = []
    for tn in test:
        test_n = tn.split('/')[-1].split('.')[0]
        test_name.append(test_n)
    test_name.sort()
    filename = '/home/yared/文件/ISBI2021/Training_Set/Training/'
    X,y = [], []
    data_gt = pd.read_csv('/home/yared/文件/ISBI2021/Training_Set/RFMiD_Training_Labels.csv')
    m,n = data_gt.shape
    for i in range(0,m):
        name = data_gt.iloc[i, 0]
        gt = data_gt.iloc[i, 2:]
        gt_array = gt.values
        gt_array = gt_array.astype(np.float)
        if str(name) not in test_name:
            X.append(filename + str(name) + '.png')
            y.append(gt_array)
    
    parameter_list = {'batch_size':3,'epochs':200}
    train_X, val_X, train_y, val_y = train_test_split(X,y,test_size=0.2,random_state=0,shuffle=True)
    #train_X = train_X + train_X
    #train_y = train_y + train_y
    
    train_loader = torch.utils.data.DataLoader(
                        dataset_TASK1(train_X, train_y), 
                        batch_size=parameter_list['batch_size'], 
                        shuffle=True, 
                        num_workers=3
                        )
    
    val_loader   = torch.utils.data.DataLoader(
                        dataset_TASK1(val_X, val_y,mode = 'val'), 
                        batch_size=parameter_list['batch_size'], 
                        shuffle=True, 
                        num_workers=3
                        )
    model = DFModel().cuda()
    # model.load_state_dict(model_path)
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.01, momentum=0.9)
    ce = AsymmetricLossOptimized()
    # # ce = DiceLoss().to(device)
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    
    resume_ind = 0    #hw
    step = 0    #hw
    find_best = []
    find_best_name=['batch','loss','F1']
    best_ep_loss = 10    #hw
    best_acc = 0    #hw
    remember_epoch = 0    #hw
    lowest_loss = 10    #hw
    lowest_loss_record = ['epoch','loss','F1']
    for epoch in range(parameter_list['epochs']): 
        ep_loss = 0.    #hw
        for batch_idx, (x,y) in enumerate(train_loader): 
            running_loss = 0.0    #hw
            optimizer.zero_grad()
            pred = model(x.to(device))
            # pred_sig = nn.functional.sigmoid(pred)
            lab = y.to(device).float()
    
            loss = ce(pred, lab)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            optimizer.second_step(zero_grad=True)
            running_loss  +=  loss.item()
            # print(batch_idx)
    
            if batch_idx % 153 == 1:  
                with torch.no_grad():
                    model.eval()
                    cnt, acc, acc2 = 0, 0., 0.
                    for ind2, (vx, vy) in enumerate(val_loader):
                        pred_prob = model(vx.to(device))
                        pred_sigv = torch.sigmoid(pred_prob)
                        # pred_sigv = pred_prob
                        pred_sigv[pred_sigv >= 0.5] = 1
                        pred_sigv[pred_sigv < 0.5] = 0
                        predv = pred_sigv.cpu().numpy()
                        v_gt = vy.cpu().numpy()
                        m1,n1 = predv.shape
                        acc_b = predv
                        for i1 in range(0,m1):
                            tp = 0
                            fp = 0
                            fn = 0
                            for j in range(0,n1):
                                if v_gt[i1,j] == 1:
                                    if predv[i1,j]==1:
                                        tp+=1
                                if v_gt[i1,j] == 1:
                                    if predv[i1,j]==0:
                                        fn+=1
                                if v_gt[i1,j] == 0:
                                    if predv[i1,j]==1:
                                        fp+=1
                            if tp != 0:
                                precision = tp/(tp+fp)
                                recall = tp/(tp+fn)
                                f_score = (2*precision*recall)/(precision + recall)
                            else:
                                f_score = 0
                            acc_b[0,i1] = f_score
                        acc += np.mean(acc_b)
                        cnt+=1
                    if epoch == 0:
                        torch.save(model.state_dict(), 'checkpoint/model_epoch_%d_1.pth' %(epoch))
    
                    print('[epoch: %d, batch: %5d] loss: %.3f, F1: %.3f' %
                          (epoch, batch_idx+resume_ind, running_loss, acc/cnt))
                    find_best.append([batch_idx+resume_ind,running_loss,acc/cnt])
                ep_loss += running_loss
                step+=1
                running_loss = 0.0
                model.train() 
                
        print('Model saved!')
        torch.save(model.state_dict(), 'checkpoint/model_epoch_%d.pth' %(epoch))
        print('Epoch averaged loss = %f' % ep_loss)
        find_best.append([epoch,'<<epoch || averaged loss>>',ep_loss])
        
        # if epoch >=10:
        #     if ep_loss < best_ep_loss and acc/cnt >= best_acc:
        #         best_ep_loss = ep_loss
        #         best_acc = acc/cnt
        #         remember_epoch = epoch
        #         print('The best!')
        #     if ep_loss < lowest_loss:
        #         lowest_loss = ep_loss
        #         lowest_loss_record.append([epoch,ep_loss,acc/cnt])
        
    pd.DataFrame(columns=find_best_name,data=find_best).to_csv('./record.csv',index=False,encoding='gbk')


