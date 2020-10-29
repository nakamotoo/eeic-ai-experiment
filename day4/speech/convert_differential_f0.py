import numpy as np
import pysptk as sptk
from scipy.io import wavfile
import os
import pyworld as pw
from pysptk.synthesis import MLSADF, Synthesizer

fs = 16000
fftlen = 512
alpha = 0.42
dim = 25

# Calculating mean/std of log_F0
datalist = []
with open("conf/train.list", "r") as f:
    for line in f:
        line = line.rstrip()
        datalist.append(line)

src_lf0 = None
tgt_lf0 = None
for i in range(0, len(datalist)):
    with open("data/SF/f0/{}.f0".format(datalist[i]), "rb") as f:
        f0 = np.fromfile(f, dtype="<f8", sep="")
    lf0 = np.log2(f0[f0 > 0.0])
    if src_lf0 is None:
        src_lf0 = lf0
    else:
        src_lf0 = np.concatenate([src_lf0, lf0])

    with open("data/TF/f0/{}.f0".format(datalist[i]), "rb") as f:
        f0 = np.fromfile(f, dtype="<f8", sep="")
    lf0 = np.log2(f0[f0 > 0.0])
    if tgt_lf0 is None:
        tgt_lf0 = lf0
    else:
        tgt_lf0 = np.concatenate([tgt_lf0, lf0])
        
src_lf0_mean = src_lf0.mean(axis=0)
src_lf0_std = src_lf0.std(axis=0)
tgt_lf0_mean = tgt_lf0.mean(axis=0)
tgt_lf0_std = tgt_lf0.std(axis=0)

# linear transformation of log F0

datalist = []
with open("conf/eval.list", "r") as f:
    for line in f:
        line = line.rstrip()
        datalist.append(line)
        
for i in range(0, len(datalist)):
    with open("data/SF/f0/{}.f0".format(datalist[i]), "rb") as f:
        f0 = np.fromfile(f, dtype="<f8", sep="")
        f0[f0 > 0.0] = np.power(2, (np.log2(f0[f0 > 0.0]) - src_lf0_mean) * tgt_lf0_std / src_lf0_std + tgt_lf0_mean)  # F0 = 0 は基本周波数が定義されていないことを意味する
    with open("data/SF-TF/f0/{}.f0".format(datalist[i]), "wb") as f:
        f0.tofile(f)

        

for i in range(0,len(datalist)):
    outfile = "result/wav/{}_diff_f0.wav".format(datalist[i])
    with open("data/SF-TF/mgc/{}.mgc".format(datalist[i]), "rb") as f:
        conv_mgc = np.fromfile(f, dtype="<f8", sep="")
        conv_mgc = conv_mgc.reshape(len(conv_mgc)//dim, dim)
    
    # f0の変換
    with open("data/SF/mgc/{}.mgc".format(datalist[i]), "rb") as f:
        src_mgc = np.fromfile(f, dtype="<f8", sep="")
        src_mgc = src_mgc.reshape(len(src_mgc)//dim, dim)
    with open("data/SF-TF/f0/{}.f0".format(datalist[i]),"rb") as f:
        f0 = np.fromfile(f, dtype="<f8", sep="")
    with open("data/SF/ap/{}.ap".format(datalist[i]),"rb") as f:
        ap = np.fromfile(f, dtype="<f8", sep="")
        ap = ap.reshape(len(ap)//(fftlen+1),fftlen+1)
    
    mgc = src_mgc.astype(np.float64)
    sp = sptk.mc2sp(mgc, alpha, fftlen*2)
    owav = pw.synthesize(f0, sp, ap, fs)
    owav = np.clip(owav, -32768, 32767)
    
#     fs, data = wavfile.read("data/SF/wav/{}.wav".format(datalist[i]))  # 入力音声そのものをもってくる
#     data = data.astype(np.float)

    diff_mgc = conv_mgc - src_mgc  # 差分のフィルタを用意する
    diff_mgc = np.zeros(shape=conv_mgc.shape)

    # 差分のフィルタを入力音声波形に適用する
    b = np.apply_along_axis(sptk.mc2b, 1, diff_mgc, alpha)
    synthesizer = Synthesizer(MLSADF(order=dim-1, alpha=alpha), 80)
    owav = synthesizer.synthesis(owav, b)
    
    owav = np.clip(owav, -32768, 32767)
    wavfile.write(outfile, fs, owav.astype(np.int16))
