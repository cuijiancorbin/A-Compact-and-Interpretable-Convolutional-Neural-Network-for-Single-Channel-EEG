# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:17:29 2019

@author: JIAN
"""
import torch
import scipy.io as sio
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
import matplotlib.gridspec as gridspec
from CompactCNN import CompactCNN

plt.rcParams.update({'font.size': 12})

torch.cuda.empty_cache()
torch.manual_seed(0)

class FeatureVis():
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_heatmap(self, allsignals,sampleidx,subid,samplelabel,multichannelsignal,likelihood):
        """
        input:
           allsignals:          all the signals in the batch
           sampleidx:           the index of the sample
           subid:               the ID of the subject
           samplelabel:         the ground truth label of the sample
           multichannelsignal:  the signals from all channels for the sample
           likelihood:          the likelihood of the sample to be classified into alert and drowsy state 
        """
        
        if likelihood[0]>likelihood[1]:
            state=0
        else:
            state=1

        if samplelabel==0:
            labelstr='alert'
        else:
            labelstr='drowsy'
                
        fig = plt.figure(figsize=(14,6))
        fig.suptitle('Subject:'+str(int(subid))+'   '+'Label:'+labelstr+'   '+'$P_{alert}=$'+str(round(likelihood[0],2))+'   $P_{drowsy}=$'+str(round(likelihood[1],2)))#, fontsize=12)        
          
# devide the figure layout                     
        gridlayout = gridspec.GridSpec(ncols=2, nrows=3, figure=fig,wspace=0.2, hspace=0.5)
        axis0 = fig.add_subplot(gridlayout[0:2,0])
        axis1 = fig.add_subplot(gridlayout[2,0])
        axis2 = fig.add_subplot(gridlayout[0:3,1])        
        
        
# do some preparations      
        
        rawsignal=allsignals[sampleidx].cpu().detach().numpy().squeeze()
        channelnum=multichannelsignal.shape[0]
        samplelength=multichannelsignal.shape[1]
        maxvalue=np.max(np.abs(rawsignal))
            
        convkernelLength=self.model.kernelLength
        
# calculate the heatmap for the sample        
        source = self.model.conv(allsignals)
        source = self.model.batch(source)
        source = torch.nn.ELU()(source)    
        
        activations=source[sampleidx].cpu().detach().numpy().squeeze()
        
        weights=self.model.fc.weight[state].cpu().detach().numpy().squeeze()
        cam=np.matmul(weights,activations)
        

        heatmap=np.zeros(samplelength)
        halfkerlength=int(convkernelLength/2)
        

        heatmap[halfkerlength:(samplelength-halfkerlength+1)]=cam
        
        for i in range(halfkerlength-1):
            heatmap[i]=heatmap[halfkerlength]*i/(halfkerlength-1)   
        for i in range((samplelength-halfkerlength),samplelength):     
            heatmap[i]=heatmap[halfkerlength]*(samplelength-halfkerlength+1-i)/(halfkerlength)
        
        heatmap= (heatmap-np.mean(heatmap)) / np.sqrt(np.sum(heatmap**2)/(samplelength))           
           
# calculate the band power components      

        psd, freqs = psd_array_multitaper(rawsignal, 128, adaptive=True,normalization='full', verbose=0)
        freq_res = freqs[1] - freqs[0]
        bandpowers=np.zeros(4)
        
        idx_band = np.logical_and(freqs >= 1, freqs <= 4)
        bandpowers[0] = simps(psd[idx_band], dx=freq_res)
        idx_band = np.logical_and(freqs >= 4, freqs <= 8)
        bandpowers[1] = simps(psd[idx_band], dx=freq_res)
        idx_band = np.logical_and(freqs >= 8, freqs <= 12)
        bandpowers[2] = simps(psd[idx_band], dx=freq_res)        
        idx_band = np.logical_and(freqs >= 12, freqs <= 30)
        bandpowers[3] = simps(psd[idx_band], dx=freq_res)


        totalpower=simps(psd, dx=freq_res)
        if totalpower<0.00000001:
            bandpowers=np.zeros(4)
        else:
            bandpowers /= totalpower
        
        
        barx=np.arange(1, 5)
        axis1.bar(barx,bandpowers)        
        axis1.set_xlim([0,5])       
        axis1.set_ylim([0,0.8])
        
        axis1.set_ylabel("Relative power")
        
        axis1.set_xticks([1,2,3,4])
        axis1.set_xticklabels(['Delta','Theta','Alpha','Beta'])

# draw the heatmap
        xx= np.arange(1, (samplelength+1))
        axis0.set_xticks([])
        axis0.set_ylim([-maxvalue-10,maxvalue+10])          
        axis0.set_xlim([0,(samplelength+1)])
        axis0.set_ylabel("mV")    
        
        points = np.array([xx, rawsignal]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = plt.Normalize(-1, 1)
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(heatmap)
        lc.set_linewidth(2)
        axis0.add_collection(lc)        
        fig.colorbar(lc,ax=axis0,orientation="horizontal", ticks=[-1, -0.5, 0, 0.5, 1])  

# draw all the signals
        
        thespan=np.percentile(multichannelsignal,98)    
        yttics=np.zeros(channelnum)
        for i in range(channelnum):
            yttics[i]=i*thespan

        axis2.set_ylim([-thespan,thespan*channelnum])          
        axis2.set_xlim([0,samplelength+1]) 

        labels=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8','T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'Oz','O2']
        
        plt.sca(axis2)
        plt.yticks(yttics, labels)
        
        heatmap1=np.zeros((channelnum,samplelength))-1 
        heatmap1[-2,:]=heatmap
        xx=np.arange(1,samplelength+1)  
           
        for i in range(0,channelnum):            
            y=multichannelsignal[i,:]+thespan*(i)
            dydx=heatmap1[i,:]
            
            points = np.array([xx, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            norm = plt.Normalize(-1, 1)
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(dydx)
            lc.set_linewidth(2)
            axis2.add_collection(lc)


def run():
    filename = r'dataset.mat'  
    
    tmp = sio.loadmat(filename)
    xdata=np.array(tmp['EEGsample'])
    label=np.array(tmp['substate'])
    subIdx=np.array(tmp['subindex'])

    label.astype(int)
    subIdx.astype(int)    
    samplenum=label.shape[0]    

    channelnum=30
    classes=2
    subjnum=11
    samplelength=3
  
    lr=1e-2# for smalle net
    sf=128
    batch_size = 50
    n_epoch =6   
    
    ydata=np.zeros(samplenum,dtype=np.longlong)
    
    for i in range(samplenum):
        ydata[i]=label[i]

    selectedchan=[28]
    rawx=xdata
    
    xdata=xdata[:,selectedchan,:]
    channelnum=len(selectedchan)

  #  you can set the subject id here
    for i in range(2,3):
        
        trainindx=np.where(subIdx!= i)[0] 
        xtrain=xdata[trainindx]   
        x_train = xtrain.reshape(xtrain.shape[0],1,channelnum, samplelength*sf)
        y_train=ydata[trainindx]
         
              
        testindx=np.where(subIdx == i)[0]    
        
        xtest=xdata[testindx]
        rawxdata=rawx[testindx]
        x_test = xtest.reshape(xtest.shape[0], 1,channelnum, samplelength*sf)
        y_test=ydata[testindx]
    
        train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        my_net = CompactCNN().double().cuda()
  
        optimizer = optim.Adam(my_net.parameters(), lr=lr)    
        loss_class = torch.nn.NLLLoss().cuda()

        for p in my_net.parameters():
            p.requires_grad = True    
            
        for epoch in range(n_epoch):   
            for j, data in enumerate(train_loader, 0):
                inputs, labels = data                
                input_data = inputs.cuda()
                class_label = labels.cuda()              
                my_net.zero_grad()               
                my_net.train()          
                class_output= my_net(input_data) 
                err_s_label = loss_class(class_output, class_label)
                err = err_s_label 
                err.backward()
                optimizer.step()

        my_net.train(False)
        with torch.no_grad():
            x_test =  torch.DoubleTensor(x_test).cuda()
            answer = my_net(x_test)
            probs=np.exp(answer.cpu().numpy()) 
            sampleVis =FeatureVis(my_net)
            
            
#   you can set the sample index here            
            sampleidx=1
            sampleVis.generate_heatmap(allsignals=x_test,sampleidx=sampleidx,subid=i,samplelabel=y_test[sampleidx],multichannelsignal=rawxdata[sampleidx],likelihood=probs[sampleidx])

if __name__ == '__main__':
    run()