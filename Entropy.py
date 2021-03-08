# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:37:41 2021

@author: Walker
"""


import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score


########################################################
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC,SVR

from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
########################################################
from scipy.signal import welch
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper


import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper

import matplotlib.gridspec as gridspec
import pywt




def bandpower(data, sf, band, method='welch', window_sec=None, relative=False):
    band = np.asarray(band)
    low, high = band
    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf
        freqs, psd = welch(data, sf, nperseg=nperseg)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                          normalization='full', verbose=0)

    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    bp = simps(psd[idx_band], dx=freq_res)
    
    return np.log(bp)

def get_power_features(tmp):
    #band power features, delta, theta, alpha, beta
    bandrange = np.array([1,4,8,12,30]) 
    features=[]
    for k in range(bandrange.shape[0]-1):
        features=features+[bandpower(tmp, 128, [bandrange[k], bandrange[k+1]], 'welch', relative=True)]
        
    return features

def get_powerEn_features(tmp):
    #band power features, delta, theta, alpha, beta
    bandrange = np.arange([1,4,8,12,30]) 
    features=[]
    for k in range(bandrange.shape[0]-1):
        features=features+[bandpower(tmp, 128, [bandrange[k], bandrange[k+1]], 'welch', relative=True)]
    
    features = np.exp(features)
    pf = features/np.sum(features)
    
    # logep = np.log2(1/pf)
    
    # spEn = np.matmul(pf,logep).tolist()
    spEn = pf.dot(np.log2(1/pf))
    
         
    return [spEn]
  


def get_wavelet_features(tmp):
    #band power features, delta, theta, alpha, beta
    features=[]
    widths = np.arange(1,64 + 1)
    coef, freqs = pywt.cwt(tmp, widths,'mexh') 
    #coef :scale(width)*timelength
    e = np.zeros(np.shape(widths))
    for i in widths-1:
        rawFeature = coef[i,:]
        e[i] = np.sqrt(rawFeature.dot(rawFeature))
    ep = e/e.sum()
    # logep = np.diag(np.log2(ep))
    
    # hp = np.matmul(ep,logep)
    hp = np.log2(ep).dot(ep)
    return [hp.sum()]

def get_approximate_entropy(x, m=2, r=0.15) -> float:
    """Approximate_entropy."""
    r = np.std(x) * r 
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(x,m):
        xmi = np.array([(x[i : i + m] -  (x[i : i + m].mean())/(m+1)) for i in range(N - m)])
        
        C1 = [(np.sum(np.abs(xmii - xmi).max(axis=1) <= r))/(N-m+1) for xmii in xmi]
        
        return (N - m + 1.0) ** (-1) * sum(np.log(C1))

    N = len(x)

    return [abs(_phi(x,m + 1) - _phi(x,m)).tolist()]


def Sample_Entropy(x, m = 2, r=0.15):
    # m:window size
    # r:threashold
    # transform x into arrary
    
    N = len(x)
    B = 0.0
    A = 0.0
    r = np.std(x) * r 
    # check x dimension
    if x.ndim != 1:
        raise ValueError("x is not 1 dimension")
    # check the length of x
    if len(x) < m+1:
        raise ValueError("len(x) less than 1")
    # split x by m window
    entropy = 0  
    
    xmi = np.array([x[i : i + m] for i in range(N - m)])
    
    B = np.sum([np.sum(np.abs(xmii - xmi).max(axis=1) <= r) - 1 for xmii in xmi])
    
    # Save all matches minus the self-match, compute B
    
    # Similar for computing A
    m += 1
    xm = np.array([x[i : i + m ] for i in range(N - m)])
    
    A = np.sum([np.sum(np.abs(xmii - xm).max(axis=1) <= r) - 1 for xmii in xm])    
    # for xmii in xm:
    #     print(xmii - xm)
    
    entropy = -np.log(A / B)

    return [entropy.tolist()]
def Fuzzy_Entropy(x, m = 2, r=0.25, n=2):
    # m:window size
    # r:threashold
    
    def _fai(xmi,m):
        
        dij = [np.abs(xmii - xmi).max(axis=1)  for xmii in xmi]
        
        dij = np.array(dij)
        dij = dij - dij[np.eye(np.shape(dij)[0],dtype = bool)]
        Dij = np.exp(-np.power(dij,n)/r)
        
        return np.sum(Dij - Dij[np.eye(np.shape(Dij)[0],dtype = bool)])/(np.shape(Dij)[0]-m -1)/(np.shape(Dij)[0] - m)
        
    

    N = len(x)
    # transform x into arrary
    x = np.array(x)
    r = np.std(x) * r 
    # check x dimension
    if x.ndim != 1:
        raise ValueError("x is not 1 dimension")
    # check the length of x
    if len(x) < m+1:
        raise ValueError("len(x) less than 1")
    m = m
    
    xmi = np.array([(x[i : i + m] -  (x[i : i + m].mean())/(m+1)) for i in range(N - m)])
    
    
     # Save all matches minus the self-match, compute fai1
    fai1 = _fai(xmi, m)

    # Similar for computing A
    m += 1
    xm = np.array([x[i : i + m] - x[i : i + m].mean()/(m+1) for i in range(N - m)])
    fai2 = _fai(xm, m)
    

    return [(np.log(fai1)-np.log(fai2)).tolist()]

def get_allfeatures(xdata,selectedchan): 
    featurespace=[]

    for i in range(xdata.shape[0]):
        features=[]
        for j in selectedchan:
            features=features+get_approximate_entropy(xdata[i,j,:])
            
        if(i%100 == 0):print("i is :",i)
        featurespace.append(features)
    xdata_out=np.array(featurespace)  
    
    return xdata_out

def run():
    filename = r'D:\DataBase\Fatigue_pipeline\journaltaiwanbalanced.mat'     

    channelnum=30
    classes=2
    subjnum=11
    samplelength=3
    
    tmp = sio.loadmat(filename)
    xdata=np.array(tmp['EEGsample'])
    label=np.array(tmp['substate'])
    subIdx=np.array(tmp['subindex'])
    
    label.astype(int)
    subIdx.astype(int)
    del tmp

    samplenum=label.shape[0]
    sf=128
    ydata=np.zeros(samplenum,dtype=np.longlong)
    
    for i in range(samplenum):
        ydata[i]=label[i]
    
    
    samplelength=3
    samplenum=label.shape[0]
    
    sf=128
    ydata=np.zeros(samplenum)
    for i in range(samplenum):
        ydata[i]=int(label[i])
    
    selectedchan= np.arange(0,channelnum-1)
    
    channelnum=len(selectedchan)

    xdata=get_allfeatures(xdata,selectedchan)
    meanacc=np.zeros(subjnum)

    for i in range(1,subjnum+1):    
        
        trainindx=np.where(subIdx != i)[0]
        testindx=np.where(subIdx == i)[0]
    
        xtrain=xdata[trainindx]     
        ytrain=ydata[trainindx]
        
        xtest=xdata[testindx]
        ytest=ydata[testindx]
        # print("xtrain is :",np.shape(xtrain))
        # print("ytrain is :",np.shape(ytrain))
    # you can select other ML methods
        classifier=SVC(gamma='scale')
       # classifier=xgboost.XGBClassifier()
      #  classifier=LogisticRegression(solver='lbfgs',max_iter=5000)
        
        classifier.fit(xtrain, ytrain)        
        predicted = classifier.predict(xtest)  
        print(accuracy_score(ytest, predicted)) 
        meanacc[i-1]=accuracy_score(ytest, predicted)
    
    print('mean')
    print(np.mean(meanacc))
    
#the corresponding name of the channels
#Fp1, Fp2, F7, F3, Fz, F4, F8, FT7, FC3, FCZ, FC4, FT8, T3, C3, Cz, C4, T4, TP7, CP3, CPz, CP4, TP8, T5, P3, PZ, P4, T6, O1, Oz  O2
#0,    1,  2,  3,  4,  5,  6,  7,   8,   9,   10,   11, 12, 13, 14, 15, 16, 17,  18,  19,  20,  21,  22,  23,24, 25, 26, 27, 28, 29, 30



if __name__ == '__main__':
    run()
    
