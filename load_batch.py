"""
BSEN
2019
Author:
        Wan-Ting Hsieh       cclee@ee.nthu.edu.tw

"""

import numpy as np
import joblib
import torch

peoID = joblib.load('data/shuffled_peoID.pkl')


def getAEBatch_centerloss(personDataList,lab_tile, batchSize=32):
    while 1:
        randDataIdx = np.random.choice(personDataList.__len__(), size=batchSize, replace=True)
        clipData = [joblib.load(personDataList[idx]) for idx in randDataIdx]
        clip_lab = [int(lab_tile[idx]) for idx in randDataIdx]
        feaAll = []

        for fea in clipData:
            if [] in fea:
                continue
            tmpFea = np.concatenate(fea, axis=1).reshape(1,fea.shape[0],fea.shape[1],fea.shape[2])
            if np.isnan(tmpFea).any() or np.isinf(tmpFea).any(): # prevent non-labe fea
                continue
            feaAll.append(tmpFea)

        feaAll = np.array(feaAll)
        feaAll = torch.from_numpy(feaAll).float().cuda()
        labAll = torch.from_numpy(np.array(clip_lab)).long().cuda()
        yield feaAll,labAll
