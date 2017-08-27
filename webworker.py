from scipy.signal import argrelextrema
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
from fbprophet import Prophet


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

#def calculate_time_series_similarity(list1,list2):
#    pear = (float(np.cov(list1,list2)[0][1])/(np.std(list1)*np.std(list2)))
##    print pear
#    MAD1 = [abs(list1[i]-list2[i]) for i in range(len(list1))]
##    print MAD1
#    MAD = sum(MAD1)/float(len(MAD1))
##    print MAD
##    print 0.5*pear+0.5/float(MAD)
#    return -(0.5*pear+0.5/float(MAD)) 

#def calculate_time_series_similarity(list1,list2):
#    
#
#    list1_max_index = list(argrelextrema(np.array(list1),np.greater)[0])
#    list1_min_index = list(argrelextrema(np.array(list1),np.less)[0])
#
#    list2_max_index = list(argrelextrema(np.array(list2),np.greater)[0])
#    list2_min_index = list(argrelextrema(np.array(list2),np.less)[0])
#
#    error=10000
#
########
##    if list1_max_index!=[] and list2_max_index!=[]:
##        error = abs(abs(list1[0]-list2[0])/list2[0])+abs(abs(list1[-1]-list2[-1])/list2[-1])+abs(abs(list1[list1_max_index[0]]-list2[list2_max_index[0]])/list2[list2_max_index[0]])+abs(1-list1_max_index[0]/list2_max_index[0])+abs(1-(len(list1)-list1_max_index[0])/(len(list2)-list2_max_index[0]))
##    if list1_min_index!=[] and list2_min_index!=[]:
##        error = abs(abs(list1[0]-list2[0])/list2[0])+abs(abs(list1[-1]-list2[-1])/list2[-1])+abs(abs(list1[list1_min_index[0]]-list2[list2_min_index[0]])/list2[list2_min_index[0]])+abs(1-list1_min_index[0]/list2_min_index[0])+abs(1-(len(list1)-list1_min_index[0])/(len(list2)-list2_min_index[0]))
#        #error = abs(list1[0]-list2[0])+abs(list1[-1]-list2[-1])+abs(list1[list1_min_index[0]]-list2[list2_min_index[0]])
########
#    if list1_max_index!=[] and list2_max_index!=[]:
##        list1 = [i/abs(list1[list1_max_index[0]]-list1[0]) for i in list1]
##        list2 = [i/abs(list2[list2_max_index[0]]-list2[0]) for i in list2]
#    #    error = abs((list2[0]/list1[0])-1)+abs((list2_max_index[0]/list1_max_index[0])-1)+abs((len(list2)-list2_max_index[0])/(len(list1)-list1_max_index[0])-1)
#        #error = abs((list2[0]-list1[0])/list2[0])+abs((list2_max_index[0]-list1_max_index[0])/list2_max_index[0])+abs(((len(list2)-list2_max_index[0])-(len(list1)-list1_max_index[0]))/(len(list2)-list2_max_index[0]))+abs((list1[list1_max_index[0]]-list2[list2_max_index[0]])/list1[list1_max_index[0]])+abs((list2[-1]-list1[-1])/list2[-1])
#        error = abs((list2[0]-list1[0])/min(list2[0],list1[0]))+abs((list2_max_index[0]-list1_max_index[0])/min(list2_max_index[0],list1_max_index[0]))+abs(((len(list2)-list2_max_index[0])-(len(list1)-list1_max_index[0]))/min((len(list2)-list2_max_index[0]),(len(list1)-list1_max_index[0])))+abs((list1[list1_max_index[0]]-list2[list2_max_index[0]])/min(list1[list1_max_index[0]],list2[list2_max_index[0]]))+abs((list2[-1]-list1[-1])/min(list1[-1],list2[-1]))+abs((len(list1)-len(list2))/min(len(list1),len(list2)))
#    if list1_min_index!=[] and list2_min_index!=[]:
##        list1 = [i/abs(list1[list1_min_index[0]]-list1[0]) for i in list1]
##        list2 = [i/abs(list2[list2_min_index[0]]-list2[0]) for i in list2]
#    #    error = abs((list2[0]/list1[0])-1)+abs((list2_min_index[0]/list1_min_index[0])-1)+abs((len(list2)-list2_min_index[0])/(len(list1)-list1_min_index[0])-1)
#        #error = abs((list2[0]-list1[0])/list2[0])+abs((list2_min_index[0]-list1_min_index[0])/list2_min_index[0])+abs(((len(list2)-list2_min_index[0])-(len(list1)-list1_min_index[0]))/(len(list2)-list2_min_index[0]))+abs((list1[list1_min_index[0]]-list2[list2_min_index[0]])/list1[list1_min_index[0]])+abs((list2[-1]-list1[-1])/list2[-1])
#        error = abs((list2[0]-list1[0])/min(list2[0],list1[0]))+abs((list2_min_index[0]-list1_min_index[0])/min(list2_min_index[0],list1_min_index[0]))+abs(((len(list2)-list2_min_index[0])-(len(list1)-list1_min_index[0]))/min((len(list2)-list2_min_index[0]),(len(list1)-list1_min_index[0])))+abs((list1[list1_min_index[0]]-list2[list2_min_index[0]])/min(list1[list1_min_index[0]],list2[list2_min_index[0]]))+abs((list2[-1]-list1[-1])/min(list1[-1],list2[-1]))+abs((len(list1)-len(list2))/min(len(list1),len(list2)))
#    return error


from sklearn.linear_model import LinearRegression
from scipy import interpolate
def __predict(imfs, sample_num, predict_num):
    r=[]
    for i in imfs:
        x = i[-sample_num:]
        rr=[]
        for j in range(0,len(i)-(sample_num+predict_num)):
            y = i[j:j+sample_num]
            rr.append([calculate_time_series_similarity(x, y),i[j+sample_num:j+(sample_num+predict_num)],[j+sample_num,j+(sample_num+predict_num)]])
        rr.sort(key=lambda x:x[0])
        r.append(np.array(rr[0][1]))
    return np.array(sum(r))

def __iter_predict(imfs, sample_num, predict_num, each_step):
    k = 10
    rrr=[]
    for i in range(len(imfs)):  
        imf = list(imfs[i])
        if i == 0:
            sn = 2
        else:
            sn = sn*2
        
        r=[]
        for step in range(int(predict_num/each_step)):
            print("sss")
            print(int(predict_num/each_step))
            imf_plus_min = np.array(imf)+abs(min(imf))+1
            
            #sn = 2*(i+1)*sample_num
            
            print("sn")
            print(sn)
            
            if sn>30:
                sn = 30
            x = imf_plus_min[-sn:]
            rr=[]
            for j in range(0,len(imf)-(sn+each_step)):
                y = imf_plus_min[j:j+sn]
                rr.append([calculate_time_series_similarity(x, y),imf[j+sn:j+(sn+each_step)],[j+sn,j+(sn+each_step)]])
            rr.sort(key=lambda x:x[0])
            print("ss")
            score = [kkk[0] for kkk in rr]
            print(score)
            score1 = np.percentile(score,5)
            print(score1)
            print([kkk[2] for kkk in rr])
            
            top_k = []
            for kk in range(k):
                if rr[kk][0]<score1:
                    top_k.append(np.array(rr[kk][1]))
            if len(top_k)==0:
                print("median")
                print(score)
                new_add = np.array([np.median(imf)]*each_step)
            else:
                print("topk")
                new_add = sum(top_k)/len(top_k)
            print("newadd {}".format(new_add.shape))
            imf.extend(new_add)
            r.extend(new_add)
            
        print("imfs")
        print(i)
        print(imf)
        rrr.append(np.array(r))
    print(rrr)
    print("end __iter")
    return np.array(sum(rrr))

def prophet_predict_imf(imfs,dates,predict_num):
    r = []
    for imf in imfs:
        plus_value = abs(min(imf))+1
        imf = np.array(imf)+plus_value
        df = pd.DataFrame({"ds":dates,"y":imf})
        y_value = list(df["y"].values)
        df['y'] = np.log(y_value)
       
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=60,include_history=False)
        forecast = m.predict(future)
        v = np.ravel(forecast[["yhat"]].values)
        print("prophet len")
        print(v)
        print(np.exp(v)-plus_value)
        r.append(np.exp(v)-plus_value)
    
    return sum(r)

def prophet_predict(data,dates,predict_num):
    print(len(data))
    print(len(dates))
    print(data)
    print(dates)
    plus_value = abs(min(data))+1
    data = np.array(data)+plus_value
    df = pd.DataFrame({"ds":dates,"y":data})
    y_value = list(df["y"].values)
    df['y'] = np.log(y_value)

    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=60,include_history=False)
    forecast = m.predict(future)
    v = np.ravel(forecast[["yhat"]].values)
    r=np.exp(v)-plus_value
    print("haha")
    print(r)
    return r

def remove_outlier(data):
    #return np.clip(data, np.percentile(data,10), np.percentile(data,90))
    mean = np.mean(data)
    std = np.std(data)
    data = np.clip(data, mean-0.2*std, mean+0.2*std)
    return np.clip(data, np.percentile(data,10), np.percentile(data,90))
    #return data
    
        

def predict_iter(data, predict_num, dates):
    try:
        raw_data = data
        #data = remove_outlier(data)
        data = list(np.diff(data))
    #    data = np.array(data)
    #    data[data==0]=0.1
    #    data = [data[i]/data[i-1] for i in range(1,len(data))]
        data.insert(0,0)
        print("data")
        print(raw_data)
        print(data)
        
        myemd=one_dimension_emd(data)
        imfs,residual = myemd.emd(0.001,0.001)
        imfs.append(residual)
        print(len(imfs))
        sample_range = [10]
        predict_step_range = [60]
        r=[]
        for sample_num in sample_range:
            print(sample_num)
            for predict_step in predict_step_range:
                print(predict_step)
                r.append(__iter_predict(imfs,sample_num,predict_num,predict_step))
        #residual_predict = np.array(splinepredict(residual,60,3))
        #residual_predict = np.median(residual[-60:])
        #pro_predict = prophet_predict_imf(imfs, dates,predict_num)
        print("end predictiter")
        first = sum(r)/(len(sample_range)*len(predict_step_range))
        print("first")
        first = np.cumsum(first)+raw_data[-1]
        print(first)
        second = [np.median(raw_data)]*predict_num
        #third = pro_predict
    
        print("first")
        print(first)
        print("second")
        print(second)
        #print("third")
        #print(third)
        #final = (first+second+third)/3
        final = (first+second)/2
        return final
    except Exception as err:
        print("error")
        print(err)
        traceback.print_exc()  
        return [np.median(data)]*predict_num




#
@app.task(serializer="pickle")
def evall(data,index,dates):
    print(data)
    print("data index:{}".format(index))
    s11=data[-250:-60]
    s111=data[:-60]
    date=dates[:-60]
    date2=dates[-250:-60]
    print("dates")
    print(date2)
    r1=predict_iter(s11,60,date2)
    try:
 
        r2=prophet_predict(s111,date,60)
    except Exception as err:
        print("error")
        print(err)
        traceback.print_exc()
        r2 = np.array([0]*len(r1))
    print("r1")
    print(r1)
    print("r2")
    print(r2)
    r = (r1+r2)/2
    print(len(r))
#    r = [np.median(s11)]*60
 
    t = data[-60:]
    p = r
    return list(r),index

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
