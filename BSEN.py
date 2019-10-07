"""
BSEN
2019
Author:
        Wan-Ting Hsieh       cclee@ee.nthu.edu.tw

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as Data
import argparse


import joblib
import warnings
import numpy as np
import os
import pandas as pd

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn import svm
import glob
import nibabel as nib
import matplotlib.pyplot as plt

from contrastive_center_loss import ContrastiveCenterLoss
from models_BSEN import AutoEncoder, ChannelModel
from load_batch import getAEBatch_centerloss

# Parsing arg
parser = argparse.ArgumentParser(description='PyTorch Center Loss Example')
parser.add_argument('--loss', type=int, default=1,
                    help='0: Center Loss, 1: Contrastive-Center Loss')
parser.add_argument('--lambda-c', type=float, default=1.0,
                    help='weight parameter of center loss (default: 1.0)')
parser.add_argument('--alpha', type=float, default=0.0001,
                    help='learning rate of class center (default: 0.5)')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num-workers', type=int, default=0,
                    help='number of workers')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# reproducibility
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# load feature's roots and labels
peoID = joblib.load('data/shuffled_peoID.pkl')
DATAROOT = 'data/brain_data_per_time/'
dataList = []
for ID in peoID:
    peoList = []
    for t in range(20,90,1):
        peoList.append(DATAROOT+ID+'_'+str(t)+'padbrain.pkl')
    dataList.append(np.concatenate((np.array(peoList),np.tile('data/brain_data/'+ID+'_'+'padbrain.pkl',10)),axis=0))
dataList = np.array(dataList)

behfile = pd.read_excel('data/Dalin_behavioral.xlsx')
CDRlab = []
MMSElab = []
MoCAlab = []
label = []
for ID in peoID:
    if ID[0]=='A':
        label.append(2)
    elif ID[0]=='M':
        label.append(1)
    elif ID[0]=='H':
        label.append(0)
    CDRlab.append(list(behfile.loc[ID])[3])   ## <0.5 -> health
    MMSElab.append(list(behfile.loc[ID])[4])  ## >26 -> health
    MoCAlab.append(list(behfile.loc[ID])[5])  ## >25 -> health

label = np.array(label)
beh_tmp = np.array(MoCAlab)
beh_lab = np.zeros((len(beh_tmp)))
beh_lab[np.where(beh_tmp>25)]=1
lab_tile = []
for i in beh_lab:
    lab_tile.append(np.tile(i,120))
lab_tile=np.array(lab_tile)


# parameter
PROJROOT = 'results/'
MODNAME = 'BSEN/'
SAVROOT = os.path.join(PROJROOT,MODNAME)

ver='MoCA'
H1 = [32]
H2 = [16]
H3 = [8]

MAXEPOCH = 30
nb_fold = 5

# training
def train(epoch,num_epochs,trainGen, model_ae, model_ct, criterion, optimizer,loss_MSE, loss_CT, loss_T):
    model_ae.train()
    model_ct.train()

    for iteration in range(int(np.ceil(dataList.shape[0]*dataList.shape[1]*(nb_fold-1)/nb_fold/args.batch_size))):
        x,y = next(trainGen)

        # forward
        output, latent = model_ae(x)
        lossMSE = criterion[0](output, x)
        lossCT = model_ct(y,latent.view(latent.shape[0], latent.shape[1]*latent.shape[2]*latent.shape[3]*latent.shape[4]))
        loss = lossMSE+0.5*lossCT

        # backward
        optimizer[0].zero_grad()
        optimizer[1].zero_grad()

        lossCT.backward(retain_graph=True)
        lossMSE.backward()
        optimizer[0].step()
        optimizer[1].step()

    loss_MSE.append(lossMSE.item())
    loss_CT.append(lossCT.item())
    loss_T.append(loss.item())

    print('epoch [{}/{}], loss_mse:{:.4f}, loss_ct:{:.4f}, loss:{:.4f}'.format(epoch + 1, num_epochs, lossMSE.item(),lossCT.item(), loss.item()))
    return model_ae,model_ct,loss_MSE, loss_CT, loss_T, optimizer


# fitting
for hidden1 in H1:
    for hidden2 in H2:
        for hidden3 in H3:
            all_result = []
            for epoch in range(0, MAXEPOCH,1):
                input_dim = hidden3*8*10*8
                print(epoch)

                #==============================================================================
                # cross validation
                #==============================================================================
                kf = KFold(n_splits=nb_fold)
                all_hiddens = []
                fold = 0
                for tr, ts in kf.split(dataList):
                    fold+=1
                    #==== prepare data =====#
                    X_train = np.hstack(dataList[tr])
                    Y_train = np.hstack(lab_tile[tr])
                    X_test = np.hstack(dataList[ts])
                    Y_test = np.hstack(lab_tile[ts])
                    trainGen = getAEBatch_centerloss(X_train,Y_train, args.batch_size)
                    mse_loss = nn.MSELoss().cuda()
                    criterion = [mse_loss]

                    #===== build or load models =====#
                    if epoch==0:
                        loss_MSE = []
                        loss_CT = []
                        loss_T = []
                        loss_all = [loss_MSE, loss_CT, loss_T]
                        model_ae = AutoEncoder(hidden1,hidden2,hidden3).cuda()
                        model_ct = ContrastiveCenterLoss(dim_hidden=input_dim, num_classes=2, lambda_c=args.lambda_c, use_cuda=args.cuda).cuda()
                        optimizer_1 = optim.Adam(list(model_ae.encoder.parameters())+list(model_ct.parameters()), lr=args.alpha, weight_decay=1e-4)
                        optimizer_2 = optim.Adam(model_ae.decoder.parameters(), lr=args.lr, weight_decay=1e-5)
                        optimizer = [optimizer_1, optimizer_2]

                        if len(X_test)==0 or len(X_train)==0: # empty check
                            continue
                    else:
                        modName = 'lr{}_ep{}_ba{}_h1{}_h2{}_h3{}'.format(str(args.lr), str(epoch-1), str(args.batch_size), str(hidden1), str(hidden2), str(hidden3))
                        model_all = joblib.load(SAVROOT+'/model/{}/{}_{}.pkl'.format(ver, str(fold), modName))
                        loss_all = joblib.load(SAVROOT+'/losses/{}/{}_{}.pkl'.format(ver, str(fold), modName))

                        model_ae = model_all[0].cuda()
                        model_ct = model_all[1].cuda()
                        optimizer = model_all[2]
                        for param in model_ae.parameters():
                            param.requires_grad = True
                        for param in model_ct.parameters():
                            param.requires_grad = True

                    model_ae,model_ct,loss_MSE,loss_CT,loss_T,optimizer = train(epoch,args.epochs,trainGen, model_ae, model_ct, criterion, optimizer,loss_all[0], loss_all[1], loss_all[2])


                    #===== save model_ae and plot loss =====#
                    model_ae.eval()
                    model_ct.eval()
                    modName = 'lr{}_ep{}_ba{}_h1{}_h2{}_h3{}'.format(str(args.lr), str(epoch), str(args.batch_size), str(hidden1), str(hidden2), str(hidden3))
                    model_all = [model_ae, model_ct, optimizer]
                    joblib.dump(model_all, SAVROOT+'model/{}/{}_{}.pkl'.format(ver, str(fold), modName))
                    joblib.dump([loss_MSE, loss_CT, loss_T], SAVROOT+'losses/{}/{}_{}.pkl'.format(ver, str(fold), modName))

                    plt.title(modName)
                    plt.plot(loss_MSE, label='mse')
                    plt.plot(loss_CT, label='ct')
                    plt.savefig(SAVROOT+'results/loss_fig/{}_{}.png'.format(ver, modName))

                    #===== Decoding =====#
                    for testidx in ts:
                        img = joblib.load('../data/brain_data/'+peoID[testidx]+'_padbrain.pkl')
                        img =torch.from_numpy(img).reshape(1,1,64,80,64).float().cuda()
                        output,hidden_extract = model_ae(img)

                        all_hiddens.append(hidden_extract.data.cpu().numpy())

                        recon_brain = model_ae.decoder(hidden_extract)
                        joblib.dump(recon_brain.data.cpu().numpy()[0][0],SAVROOT+'recon_brain/{}/{}_{}_recon_brain.pkl'.format(ver,peoID[testidx], modName))

                joblib.dump(all_hiddens,SAVROOT+'results/CT_{}_{}.pkl'.format(ver, modName))


                #===== Evaluation =====#
                all_data = []
                for i in all_hiddens:
                    tmp = i.reshape(i.shape[1],i.shape[2]*i.shape[3]*i.shape[4]) #8,8*10*8
                    all_data.append(np.mean(tmp,axis=0))

                feature = np.array(all_data)

                kf = KFold(n_splits=nb_fold)

                report = []
                for p in range(10,110,10):
                    warnings.filterwarnings("ignore")
                    true_lab = []
                    all_pred = []
                    selectedFeat = []
                    pvals = []
                    scores = []
                    for tr, ts in kf.split(feature):
                        X_train = np.vstack(feature[tr])
                        Y_train = np.hstack(label[tr])
                        X_test = np.vstack(feature[ts])
                        Y_test = np.hstack(label[ts])
                        true_lab.append(Y_test)

                        fs = SelectPercentile(score_func = f_classif, percentile = p).fit(X_train, Y_train)
                        Train_data = fs.transform(X_train)
                        Test_data = fs.transform(X_test)

                        clf = svm.SVC(C = 1, kernel='linear',class_weight = 'balanced')
                        clf = clf.fit(Train_data, Y_train)
                        predTest = clf.predict(Test_data)
                        all_pred.append(predTest)
                    pred = np.hstack(all_pred)
                    true_lab = np.hstack(true_lab)
                    conf = confusion_matrix(true_lab,pred)
                    uar = recall_score(true_lab,pred, average='macro')


                    report.append([uar, conf, p])
                    report.sort(key=lambda tup: tup[0], reverse=True)

                print("========================== epoch %i ============================" %epoch)
                print(SAVROOT+'results/CT_{}_{}.pkl'.format(ver, modName))
                print(report[0])
                all_result.append(report[0][0])

            print('best result: %f at epoch%i' %(max(all_result),np.argmax(all_result)) )
