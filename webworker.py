
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from emd import one_dimension_emd
import time
import numpy as np
from multiprocessing import Pool
from celery import Celery
import time
from celery import group
from celery.result import allow_join_result
from sklearn import preprocessing
from numba import jit
import traceback


broker = 'redis://127.0.0.1:6379'
backend = 'redis://127.0.0.1:6379'


CELERY_ACCEPT_CONTENT = ['pickle', 'json', 'msgpack', 'yaml']
CELERY_RESULT_SERIALIZER = 'pickle'


app = Celery('tasks', broker=broker, backend=backend, task_serializer='pickle')
app.conf.update(CELERY_ACCEPT_CONTENT = ['pickle', 'json', 'msgpack', 'yaml'], CELERY_RESULT_SERIALIZER = 'pickle')


@jit
def calculate_time_series_similarity(x, y):

    total = 0
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            temp = np.log(max((x[j]/x[i])/(y[j]/y[i]),(y[j]/y[i])/(x[j]/x[i])))
            total += temp
    result= np.exp(total/(len(x)**2))-1
    return result

from sklearn.linear_model import LinearRegression
from scipy import interpolate
def __predict(imfs, sample_num, predict_num):
    r=[]
    for i in imfs:
        x = i[-sample_num:]
        rr=[]
        for j in range(0,len(i)-(sample_num+predict_num)):
            y = i[j:j+sample_num]
            rr.append([calculate_time_series_similarity(x, y),i[j+sample_num:j+(sample_num+predict_num)]])
        rr.sort(key=lambda x:x[0])
        r.append(np.array(rr[0][1]))
    return np.array(sum(r))

def __iter_predict(imfs, sample_num, predict_num, each_step):
    k = 3
    rrr=[]
    for i in range(len(imfs)):  
        imf = list(imfs[i])
        
        r=[]
        for step in range(int(predict_num/each_step)):
            print("sss")
            print(int(predict_num/each_step))
            imf_plus_min = np.array(imf)+abs(min(imf))+1
            
            sn = 2*(i+1)*sample_num
            print("sn")
            print(sn)
            
            if sn>60:
                sn = 60
            x = imf_plus_min[-sn:]
            rr=[]
            for j in range(0,len(imf)-(sn+each_step)):
                y = imf_plus_min[j:j+sn]
                rr.append([calculate_time_series_similarity(x, y),imf[j+sn:j+(sn+each_step)]])
            rr.sort(key=lambda x:x[0])
            print("ss")
            print([kkk[0] for kkk in rr])
            top_k = []
            for kk in range(k):
                top_k.append(np.array(rr[-kk][1]))
            new_add = sum(top_k)/k
            imf.extend(new_add)
            r.extend(new_add)
            
        rrr.append(np.array(r))
    print(rrr)
    print("end __iter")
    return np.array(sum(rrr))

def predict_iter(data, predict_num):
    try:
        myemd=one_dimension_emd(data)
        imfs,residual = myemd.emd(0.01,0.01)
        imfs.append(residual)
        print(len(imfs))
        sample_range = [2]
        predict_step_range = [1]
        r=[]
        for sample_num in sample_range:
            print(sample_num)
            for predict_step in predict_step_range:
                print(predict_step)
                r.append(__iter_predict(imfs,sample_num,predict_num,predict_step))
        #residual_predict = np.array(splinepredict(residual,60,3))
        #residual_predict = np.median(residual[-60:])
        print("end predictiter")
        first = sum(r)/(len(sample_range)*len(predict_step_range))
        second = np.median(data)
    
        final = (first+second)/2
        return final
    except Exception as err:
        print(err)
        traceback.print_exc()  
        return [np.median(data)]*predict_num




#
@app.task(serializer="pickle")
def evall(data,index):
    try:
        print(data)
        print("data index:{}".format(index))
        s11=data[:-60]
        r=predict_iter(s11,60)
        print(r)

        t = data[-60:]
        p = r
        return list(r),index
    except Exception as err:
        print(err)
        traceback.print_exc()  
        return [],index

def smape(target,pred):
    up = abs(target-pred)
    down = (abs(target)+abs(pred))/2
    #down = (target+pred)/2
    return sum(up/down)/len(target)

import math

@jit
def smape_fast(y_true, y_pred):
    out = 0
    for i in range(y_true.shape[0]):
        a = y_true[i]
        b = y_pred[i]
        c = a+b
        if c == 0:
            continue
        out += math.fabs(a - b) / c
    out *= (200.0 / y_true.shape[0])
    return out

#s1 = train_df.loc[128447].values[-150:]
#print(s1)
#eval(s1)
