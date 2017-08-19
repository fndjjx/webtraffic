
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
from webworker import evall


#broker = 'redis://127.0.0.1:6379'
#backend = 'redis://127.0.0.1:6379'


#app = Celery('tasks', broker=broker, backend=backend, task_serializer='pickle')
#app.conf.update(CELERY_ACCEPT_CONTENT = ['pickle'], CELERY_RESULT_SERIALIZER = 'pickle')

train_df = pd.read_csv("/tmp/train_1.csv")
train_df = train_df.fillna(1)
#key_df = pd.read_csv("/tmp/key_1.csv")
#sample_df = pd.read_csv("/tmp/sample_submission_1.csv")

def calculate_time_series_similarity(x, y):
    total = 0
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            temp = np.log(max((x[j]/x[i])/(y[j]/y[i]),(y[j]/y[i])/(x[j]/x[i])))
            total += temp
    return np.exp(total/(len(x)**2))-1

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
    rrr=[]
    for i in range(len(imfs)):  
        imf = list(imfs[i])
        
        r=[]
        for step in range(int(predict_num/each_step)):
            
            sn = 2*(i+1)*sample_num
            
            if sn>100:
                sn = 100
            x = imf[-sn:]
            rr=[]
            for j in range(0,len(imf)-(sn+each_step)):
                y = imf[j:j+sn]
                rr.append([calculate_time_series_similarity(x, y),imf[j+sn:j+(sn+each_step)]])
            rr.sort(key=lambda x:x[0])
            imf.extend(rr[0][1])
            r.extend(rr[0][1])
            
        rrr.append(np.array(r))
    return np.array(sum(rrr))

def predict_iter(data, predict_num):
    myemd=one_dimension_emd(data)
    imfs,residual = myemd.emd(0.01,0.01)
    imfs.append(residual)
    print(len(imfs))
    sample_range = [1]
    predict_step_range = [10,30,60,120]
    r=[]
    for sample_num in sample_range:
        print(sample_num)
        for predict_step in predict_step_range:
            print(predict_step)
            r.append(__iter_predict(imfs,sample_num,predict_num,predict_step))
    #residual_predict = np.array(splinepredict(residual,60,3))
    #residual_predict = np.median(residual[-60:])
    return (sum(r)/(len(sample_range)*len(predict_step_range)))

def predict(data, predict_num):
    myemd=one_dimension_emd(data)
    imfs,residual = myemd.emd(0.01,0.01)
    imfs.append(residual)
    print(len(imfs))
    sample_range = [30]
    r=[]
    for sample_num in sample_range:
        r.append(__predict(imfs,sample_num,predict_num))
    return sum(r)/len(sample_range)

def eval_iter():
    result = []
    index = []
    for i in range(10):
        try:
            rn = np.random.randint(0,140000)
            s1 = train_df.loc[rn].values[-150:]
            result.append(evall(s1))
            index.append(rn)
        except:
            pass
    print(result)
    print(index)
    print(len(result))
    print(len(index))
    result = list(filter(lambda x:str(x)!="nan",result))
    print(np.mean(result))
    print(np.std(result))

def eval_iter_multi():
    result = []
    index = []
    pool = Pool(processes=4)
    para = [train_df.loc[np.random.randint(0,140000)].values[-150:] for i in range(1000)]
    result=pool.map(evall, para)
    print(result)
    print(len(result))
    result = list(filter(lambda x:str(x)!="nan",result))
    print("result")
    print(np.mean(result))
    print(np.std(result))

def eval_iter_multi2(n):
    para = [train_df.loc[np.random.randint(0,140000)].values[-150:] for i in range(n)]
    print(para)
    tasks = [evall.signature((para[p_index],p_index)) for p_index in range(len(para))]
    result_group = group(tasks)()
    with allow_join_result():
        result = result_group.get()

    p = []
    t = []
    for i in result:
        if len(i[0])!=0:
            p.extend(i[0])
            t.extend(para[i[1]][-60:])
    
    return p,t

def eval_loop(total_n,each_n):
    ln = total_n//each_n
    last_each_n = total_n%each_n
    print(ln)
    print(last_each_n)

    p = []
    t = []
    for i in range(ln+1):
        if i!=ln:
            pp,tt=eval_iter_multi2(each_n)
            p.extend(pp)
            t.extend(tt)
        else:
            pp,tt=eval_iter_multi2(last_each_n)
            p.extend(pp)
            t.extend(tt)

    print("result")
    print(len(p))
    print(len(t))
    print(smape_fast(np.array(t),np.array(p)))

def eval_loop2(base,n):
    para = [train_df.loc[base+i].values[-150:] for i in range(n)]
    print(para)
    tasks = [evall.signature((para[p_index],p_index)) for p_index in range(len(para))]
    result_group = group(tasks)()
    with allow_join_result():
        result = result_group.get()

    p = []
    t = []
    for i in result:
        if len(i[0])!=0:
            p.extend(i[0])
            t.extend(para[i[1]][-60:])

    print("result")
    print(len(p))
    print(len(t))
    print(smape_fast(np.array(t),np.array(p)))

    


def smape(target,pred):
    up = abs(target-pred)
    down = (abs(target)+abs(pred))/2
    #down = (target+pred)/2
    return sum(up/down)/len(target)

from numba import jit
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
t0 = time.time()
#eval_iter_multi2(100)
eval_loop2(5000,1000)
t1 = time.time()
print("time")
print(t1-t0)
