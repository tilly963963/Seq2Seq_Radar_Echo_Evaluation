## Others lib
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from datetime import timedelta

## ML lib
import tensorflow as tf
import keras.backend as K

## Env setting
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
K.set_session(sess)

date = '20200527_0000'
# save_path = 'CSI_PICTURE/Compare_{}/'.format(date)
save_path = 'Compare_{}/'.format(date)

if not os.path.isdir(save_path):
    os.makedirs(save_path)

csi_predrnn_loss_atten = np.loadtxt(open(save_path+"202005270000to6_predrnn_loss_atten.csv","rb"), delimiter=",", skiprows=0)
csi_CREF = np.loadtxt(open(save_path+"202005270000to6_cref.csv","rb"), delimiter=",", skiprows=0)
# csi_MLC = np.loadtxt(open("MLC_{}/csi.csv","rb"), delimiter=",", skiprows=0)
csi_PredRNN = np.loadtxt(open(save_path+"202005270000to6_predrnn_loss_atten_v2.csv","rb"), delimiter=",", skiprows=0)
csi_PredRNN_loss = np.loadtxt(open(save_path+"202005270000to6_predrnn_loss.csv","rb"), delimiter=",", skiprows=0)
csi_predrnn_atten = np.loadtxt(open(save_path+"202005270000to6_predrnn_atten.csv","rb"), delimiter=",", skiprows=0)

csi_ConvLSTM = np.loadtxt(open(save_path+"202005270000to6_convlstm.csv","rb"), delimiter=",", skiprows=0)
csi_ConvLSTM_loss = np.loadtxt(open(save_path+"202005270000to6_convlstm_loss.csv","rb"), delimiter=",", skiprows=0)

print("np.array(csi_predrnn_loss_atten).shape",np.array(csi_predrnn_loss_atten).shape)#np.array(csi).shape (6, 60)
print("np.array(csi_CREF).shape",np.array(csi_CREF).shape)
# print("np.array(csi_MLC).shape",np.array(csi_MLC).shape)
print("np.array(csi_PredRNN).shape",np.array(csi_PredRNN).shape)
print("np.array(csi_PredRNN_loss).shape",np.array(csi_PredRNN_loss).shape)

# Draw thesholds AVG CSI
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
ax.set_facecolor((229.0/255.0, 229.0/255.0, 229.0/255.0))
plt.xlim(0, 60)
plt.ylim(-0.05, 1.0)
plt.xlabel('Threshold')
plt.ylabel('CSI')
plt.title('{}'.format(date))
plt.grid(True)

all_csi = []
plt.plot(np.arange(csi_CREF.shape[1]), [np.nan] + np.mean(csi_CREF[:, 1:], 0).tolist(), c='r', linewidth=2.0, label='OP')
plt.plot(np.arange(csi_predrnn_loss_atten.shape[1]), [np.nan] + np.mean(csi_predrnn_loss_atten[:, 1:], 0).tolist(), c='b', linewidth=2.0, label='PredRNN_weightedloss_atten')
plt.plot(np.arange(csi_PredRNN.shape[1]), [np.nan] + np.mean(csi_PredRNN[:, 1:], 0).tolist(), c='chocolate', linewidth=2.0, label='PredRNN_weightedloss_atten_Transfer')
# plt.plot(np.arange(csi_ConvLSTM.shape[1]), [np.nan] + np.mean(csi_ConvLSTM[:, 1:], 0).tolist(), c='pink', linewidth=2.0, label='ConvLSTM')
# plt.plot(np.arange(csi_predrnn_atten.shape[1]), [np.nan] + np.mean(csi_predrnn_atten[:, 1:], 0).tolist(), c='g', linewidth=2.0, label='PredRNN_atten')
# plt.plot(np.arange(csi_ConvLSTM_loss.shape[1]), [np.nan] + np.mean(csi_ConvLSTM_loss[:, 1:], 0).tolist(), c='m', linewidth=2.0, label='ConvLSTM_weightedloss')

# plt.plot(np.arange(csi_PredRNN_loss.shape[1]), [np.nan] + np.mean(csi_PredRNN_loss[:, 1:], 0).tolist(), c='c', linewidth=2.0, label='PredRNN_weightedloss')
# plt.plot(np.arange(csi_MLC.shape[1]), [np.nan] + np.mean(csi_MLC[:, 1:], 0).tolist(), c='y', linewidth=3.0, label='MLC-LSTM CSI')

plt.legend(loc='upper right')

fig.savefig(fname=save_path+'Thresholds_AVG_CSI_predenn_convlstm_527.png', format='png')
plt.clf()
