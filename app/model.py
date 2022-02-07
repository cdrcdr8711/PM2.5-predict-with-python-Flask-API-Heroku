# -*- coding: UTF-8 -*-
import pickle
import gzip

# 載入模型
with gzip.open('app/model/XGB_for_PM2.5_chengdu.pgz', 'rb') as f:
    xgboostModel = pickle.load(f)

# 將模型預測寫成一個 function


def predict(input):
    pred = (xgboostModel.predict(input)[0])**3
    return pred
