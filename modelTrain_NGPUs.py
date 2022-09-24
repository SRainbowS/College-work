'''
In this example, the CSI feedback based on received pilot signal is solved step by step
  - step-1: Train an AI model for channel estimation
  - step-2: Calculate eigenvectors in subband level based on estimated channel
  - step-3: Train an AI model (encoder + decoder) for CSI compression and reconstruction
'''
#=======================================================================================================================
#=======================================================================================================================
# Package Importing
import os
import time
import gc
import numpy as np
import torch
import torch.nn as nn
from modelDesign import *
import datetime
#=======================================================================================================================
#=======================================================================================================================
# Parameters Setting
NUM_FEEDBACK_BITS = 64
NUM_RX = 4
NUM_TX = 32
NUM_SUBBAND = 13
BATCH_SIZE = 1024
EPOCHS = 100
LEARNING_RATE = 6e-5

modelSubmit_path = './modelSubmit/'
model_ce_path = './modelSubmit/encModel_p1_1.pth.tar'
model_encoder_path = './modelSubmit/encModel_p1_2.pth.tar'
model_decoder_path = './modelSubmit/decModel_p1_1.pth.tar'
if not os.path.exists(modelSubmit_path):
    os.makedirs(modelSubmit_path)

#=======================================================================================================================
#=======================================================================================================================

# Data Loading
# load pilot
pilot = np.load('../Data/pilot_1.npz')['data']
pilot = np.expand_dims(pilot, axis=1)
pilot = np.concatenate([np.real(pilot), np.imag(pilot)], 1)             # 实部虚部拼接

# load eigenvectors  特征向量
w = np.load('../Data/w.npz')['data']
w = np.expand_dims(w, axis=1)
w = np.concatenate([np.real(w), np.imag(w)], 1)

# Load time domain waveform  加载时域波形
# ht = np.load('../Data/h_time.npz')['data']
# print("pilot.shape:", pilot.shape, "\nw.shape:", w.shape, "\nht.shape:", ht.shape)
print("pilot.shape:", pilot.shape, "\nw.shape:", w.shape)
print('data loading is finished ...\t', datetime.datetime.now())

'''
# select part of samples for test
sample_num = 10000
pilot, w, ht = pilot[:sample_num,...], w[:sample_num,...], ht[:sample_num,...]
print(pilot.shape, w.shape, ht.shape)
'''

#=======================================================================================================================
#=======================================================================================================================
# Generate Label for Channel Estimation

# following is an example of generating label of channel estimation (transform channel from time to frequency domain)
# subcarrierNum = 52 * 12  # 52 resource block, 12 subcarrier per resource block
# estimatedSubcarrier = np.arange(0, subcarrierNum, 12)    #  configure the channel on which subcarriers to be estimated (this example only estimate channel on part subcarriers)
# FFTsize = 1024
# delayNum = ht.shape[-1]
# batch_size = 300
# ht_batch = ht.reshape(-1, batch_size, ht.shape[1], ht.shape[2], ht.shape[3])
# hf_batch = np.zeros([ht_batch.shape[0], batch_size, ht.shape[1], ht.shape[2], len(estimatedSubcarrier)], dtype='complex64')
# for i in range(ht_batch.shape[0]):
#     if i % 100 == 0 or i == ht_batch.shape[0]-1:
#         print("CE Label Generation Progress: {}/{}".format(i+1, ht_batch.shape[0]), flush=True)
#     ht_ = ht_batch[i, ...]
#     # padding
#     ht_ = np.pad(ht_, ((0, 0), (0, 0), (0, 0), (0, FFTsize-delayNum)))
#     # FFT
#     hf_ = np.fft.fftshift(np.fft.fft(ht_), axes=(3,))
#     startSample, endSample = int(FFTsize/2-subcarrierNum/2), int(FFTsize/2+subcarrierNum/2)
#     hf_all_subc = hf_[..., startSample:endSample]  # channel on all subcarrier
#     hf_batch[i,...] = hf_all_subc[..., estimatedSubcarrier]  # save channel on selected subcarrier
# print("hf_batch.shape:", hf_batch.shape, "\thf_batch.dtype:", hf_batch.dtype)
# hf = hf_batch.reshape(-1, hf_batch.shape[2], hf_batch.shape[3], hf_batch.shape[4])
# del ht, ht_batch, hf_batch
# gc.collect()
hf = np.load('../Data/hf.npz')['data']
hf = np.expand_dims(hf, axis=1)
hf = np.concatenate([np.real(hf), np.imag(hf)], 1)
print("hf.shape:", hf.shape)
print('label generation for channel estimation is finished ...')            # 信道估计的标签生成

#=======================================================================================================================
#=======================================================================================================================
# Channel Estimation Model Training and Saving    信道估计模型训练与保存
subc_num = pilot.shape[3]               # 208
# 设置多卡计算
device_ids = [0, 1, 2, 3, 4, 5] # 可用GPU
model_ce = channel_est(subc_num)
model_ce = torch.nn.DataParallel(model_ce, device_ids=device_ids)
model_ce = model_ce.cuda(device=device_ids[0])

criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model_ce.parameters(), lr=LEARNING_RATE)
# data loading
split_idx = int(0.95 * pilot.shape[0])
pilot_train, pilot_test = pilot[:split_idx, ...], pilot[split_idx:, ...]
hf_train, hf_test = hf[:split_idx, ...], hf[split_idx:, ...]
del hf
gc.collect()

train_dataset = DatasetFolder_mixup(pilot_train, hf_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE*len(device_ids), shuffle=True, num_workers=8, pin_memory=True)
test_dataset = DatasetFolder(pilot_test, hf_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE*len(device_ids), shuffle=False, num_workers=8, pin_memory=True)

# model training and saving
bestLoss = 10
for epoch in range(EPOCHS):
    start = time.time()
    model_ce.train()
    trainLoss = 0
    for i, (modelInput, label) in enumerate(train_loader):
        modelInput, label = modelInput.cuda(device=device_ids[0]), label.cuda(device=device_ids[0])
        modelOutput = model_ce(modelInput)
        loss = criterion(label, modelOutput)
        trainLoss += loss.item() * modelInput.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avgTrainLoss = trainLoss / len(train_dataset)
    model_ce.eval()
    testLoss = 0
    with torch.no_grad():
        for i, (modelInput, label) in enumerate(test_loader):
            modelInput, label = modelInput.cuda(device=device_ids[0]), label.cuda(device=device_ids[0])
            modelOutput = model_ce(modelInput)
            testLoss += criterion(label, modelOutput).item() * modelInput.size(0)
            testLoss += loss.item() * modelInput.size(0)
        avgTestLoss = testLoss / len(test_dataset)
        if epoch % 10 == 0:
            print('Epoch:[{0}]\t' 'Train Loss:{loss1:.5f}\t' 'Val Loss:{loss2:.5f}\t' 'Time:{time:.1f}secs\t'.format(epoch, loss1=avgTrainLoss, loss2=avgTestLoss, time=time.time()-start), datetime.datetime.now())
        if avgTestLoss < bestLoss:
            torch.save({'state_dict': model_ce.state_dict(), }, model_ce_path)
            bestLoss = avgTestLoss
            print("Model saved")
# del hf
# gc.collect()
print('model training for channel estimation is finished ...\t', datetime.datetime.now())          # 信道估计模型训练完成
#=======================================================================================================================
#=======================================================================================================================
# Calculating Eigenvectors based on Estimated Channel

def cal_eigenvector(channel):
    """
        Description:
            calculate the eigenvector on each subband
        Input:
            channel: np.array, channel in frequency domain,  shape [batch_size, rx_num, tx_num, subcarrier_num]
        Output:
            eigenvectors:  np.array, eigenvector for each subband, shape [batch_size, tx_num, subband_num]
    """
    subband_num = 13
    hf_ = np.transpose(channel, [0,3,1,2])  # (batch,subcarrier,4,32)
    hf_h = np.conj(np.transpose(channel, [0,3,2,1]))  # (batch,subcarrier,32,4)
    R = np.matmul(hf_h, hf_)  # (batch,subcarrier,32,32)
    R = R.reshape(R.shape[0], subband_num, -1, R.shape[2], R.shape[3]).mean(axis=2)  # average the R over each subband, shape:(batch,subband,32,32)
    [D,V] = np.linalg.eig(R) 
    v = V[:,:,:,0]
    eigenvectors = np.transpose(v,[0,2,1])
    return eigenvectors

w_pre = []
model_ce.load_state_dict(torch.load(model_ce_path)['state_dict'])
model_ce.eval()
test_dataset = DatasetFolder_eval(pilot)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
with torch.no_grad():
    for idx, data in enumerate(test_loader):
        print("Eigenvectors Calculation Progress: {}/{}".format(idx+1, len(test_loader)), end="\r", flush=True)
        data = data.cuda()
        # step 1: channel estimation
        h = model_ce(data)  # (batch,2,4,32,52)
        # step 2: eigenvector calculation
        h_complex = h[:,0,...] + 1j*h[:,1,...]  # (batch,4,32,52)
        h_complex = h_complex.cpu().numpy()
        v = cal_eigenvector(h_complex)
        w_complex = torch.from_numpy(v)
        w_tmp = torch.zeros([h.shape[0], 2, 32, 13], dtype=torch.float32).cuda()  # (batch,2,32,13)
        w_tmp[:,0,:,:] = torch.real(w_complex)
        w_tmp[:,1,:,:] = torch.imag(w_complex)
        w_tmp = w_tmp.cpu().numpy()
        if idx == 0:
            w_pre = w_tmp
        else:
            w_pre = np.concatenate((w_pre, w_tmp), axis=0)
del pilot
gc.collect()
print("\nw_pre.shape:", w_pre.shape)
print('eigenvectors calculation based on estimated channel is finished ...\t', datetime.datetime.now())
#=======================================================================================================================
#=======================================================================================================================
# CSI feedback Model Training and Saving   CSI反馈模型培训与保存

model_fb = AutoEncoder(NUM_FEEDBACK_BITS)
model_fb = torch.nn.DataParallel(model_fb, device_ids=device_ids)
model_fb = model_fb.cuda(device=device_ids[0])

criterion = CosineSimilarityLoss().cuda()
optimizer = torch.optim.Adam(model_fb.parameters(), lr=LEARNING_RATE)
# data loading
split_idx = int(0.95 * w.shape[0])
wpre_train, wpre_test = w_pre[:split_idx,...], w_pre[split_idx:,...]
w_train, w_test = w[:split_idx,...], w[split_idx:,...]
train_dataset = DatasetFolder_mixup(wpre_train, w_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE*len(device_ids), shuffle=True, num_workers=8, pin_memory=True)
test_dataset = DatasetFolder(wpre_test, w_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE*len(device_ids), shuffle=False, num_workers=8, pin_memory=True)
bestLoss = 10
for epoch in range(EPOCHS):
    start = time.time()
    model_fb.train()
    trainLoss = 0
    for i, (modelInput, label) in enumerate(train_loader):
        modelInput, label = modelInput.cuda(device=device_ids[0]), label.cuda(device=device_ids[0])
        modelOutput = model_fb(modelInput)
        loss = criterion(label, modelOutput)
        trainLoss += loss.item() * modelInput.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avgTrainLoss = trainLoss / len(train_dataset)
    model_fb.eval()
    testLoss = 0
    with torch.no_grad():
        for i, (modelInput, label) in enumerate(test_loader):
            modelInput, label = modelInput.cuda(device=device_ids[0]), label.cuda(device=device_ids[0])
            modelOutput = model_fb(modelInput)
            testLoss += criterion(label, modelOutput).item() * modelInput.size(0)
        avgTestLoss = testLoss / len(test_dataset)
        if epoch % 10 == 0:
            print('Epoch:[{0}]\t' 'Train Loss:{loss1:.5f}\t' 'Val Loss:{loss2:.5f}\t' 'Time:{time:.1f}secs\t'.format(epoch, loss1=avgTrainLoss, loss2=avgTestLoss, time=time.time()-start), datetime.datetime.now())
        if avgTestLoss < bestLoss:
            # Model saving
            # Encoder Saving
            torch.save({'state_dict': model_fb.encoder.state_dict(), }, model_encoder_path)
            # Decoder Saving
            torch.save({'state_dict': model_fb.decoder.state_dict(), }, model_decoder_path)
            print("Model saved")
            bestLoss = avgTestLoss
print('model training for CSI feedback is finished ...')
print('Training is finished!\t', datetime.datetime.now())

# os.system('shutdown /s /t 60')
