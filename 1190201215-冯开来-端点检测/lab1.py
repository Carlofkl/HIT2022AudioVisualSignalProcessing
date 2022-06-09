'''
选择一款语音编辑和处理工作（推荐CoolEdit pro），使用该工具，打开并显示“1.wav”的波形文件，拷屏贴图到实验报告中.
查看该语音的语谱图，拷屏贴图到实验报告中.
通过听辩，在当前窗口显示该语音信号中第一个音节的时域波形，拷屏贴图到实验报告中.
在实验报告中记录该语音数据的“采样频率”、“量化比特数”和“声道个数”等参数（注:显示在屏幕的右下角）


逐一检查各实验语料（共10个wav文件）
编写从wav文件中读入数据的程序。wav文件有44个byte的文件头，可直接跳过，后面就是各采样点，按序排列，每个占用两个byte。
逐帧读入语音数据，帧长设置为256个采样点，无帧叠。根据公式计算每帧的能量和过零率值，将其存储在数组中，并保存在文本文件中。
每个语料文件生成一个能量文件和一个过零率文件，
文件名为语料文件名加”_en.txt”和“_zero.txt”,如对”1.wav”语料生成“1_en.txt”和“1_zero.txt”。
在文本文件中，每一帧的能量（过零率）值占1行。
提交这些特征文本文件（电子版）作为实验报告的附件。

设计端点检测算法：
提供如下两种思路：（也可以采用其它方法和策略）
1）找语音部分，可采用双门限法（见课程PPT）
2）找背音部分，能量小于某一门限，且连续持续若干帧。
所采用的门限应该是自适应的，如前50帧平均值的1.1倍等
算法对数据进行判定，确定各帧的标签（语音/静音）
根据标签，生成新语料文件，只包含语料的部分。新的语料文件为RAW格式，采用“.pcm”为文件后缀。不包含文件头，直接按序存储各采样点。
提交各新语料文件作为实验报告附件。


打开新生成的语料文件进行听辩，（注：由于是RAW格式，必须正确设置采样频率、量化bit数，声道数等参数。）
根据如下信息判断是否检测正确
1）是否已经没有残留的静音？
2）语音内容保留的是否完整？
在实验报告中记录正确的文件数目
'''

import os
import wave
import numpy as np
import matplotlib.pylab as plt

# 读取文件
def getWaveData(file_path):
    f = wave.open(file_path, "rb")
    params = f.getparams()
    # 声道数，量化位数，采样频率，采样点数
    nchannels, sampwidth, framerate, nframes = params[:4]
    # print(nchannels, sampwidth, framerate, nframes)
    # 得到每个采样点的值
    str_data = f.readframes(nframes)
    f.close()
    # 转成short类型
    wave_data = np.frombuffer(str_data, dtype=np.short)
    time = np.arange(0, nframes) / framerate
    # print(nframes)
    # 归一化处理
    # wave_data = wave_data * 1.0 / (max(abs(wave_data)))
    # print(wave_data[:10])
    # 通过采样点数和取样频率计算每个取样的时间
    return wave_data, time, nchannels, sampwidth, framerate

# 分帧
def findSegment(wave_data, wlen=256, inc=256):
    # 无帧叠，帧长等于帧移，默认值为256
    signal_len = len(wave_data) # 信号总长度
    fn = int(np.ceil((signal_len) / wlen))  # 帧数
    # 可能存在帧数x采样点数量，计算得到实际长度大于本身信号长度，补零
    zeros = np.zeros(fn * wlen - signal_len)
    wave_data_zero = np.concatenate((wave_data, zeros))
    # 给每个采样点加上标签，没用
    # indices = np.tile(np.arange(0, wlen), (fn, 1)) + np.tile(np.arange(0, fn*inc, inc), (wlen, 1)).T
    # print(indices[171:172])
    # indices = np.array(indices, dtype=np.int32)
    # frames = wave_data_zero[indices]
    # 信号分帧，fn * wlen
    frames = np.array(wave_data_zero).reshape(fn, wlen)
    # print(frames)
    # window = np.hamming(wlen) # 调用海明窗
    # frames = frames * window # 信号加窗
    # for i in range(fn):
    #     frames[i] = frames[i] * window
    return frames

# 计算每一帧能量
def compute_en(frames, i):
    fn, wlen = frames.shape
    en = np.zeros((fn, 1))
    for n in range(fn):
        en[n] = np.dot(frames[n], frames[n].T)
    np.savetxt("能量\\" + str(i+1) + "_en.txt", en, fmt="%.2f")
    return en

# 计算每一帧过零率
def compute_zcr(frames, i):
    fn, wlen = frames.shape
    zcr = np.zeros((fn, 1))
    for m in range(fn):
        for n in range(wlen-1):
            # zcr[m] += np.abs(sgn(frames[m, n+1]) - sgn(frames[m, n]))
            if (frames[m, n] < 0 and frames[m, n+1] >=0) or (frames[m, n] >= 0 and frames[m, n+1] < 0):
                zcr[m] += 2
            else:
                zcr[m] += 0
        zcr[m] /= (2*wlen)
    np.savetxt("过零率\\" + str(i + 1) + "_zero.txt", zcr, fmt="%.8f")
    return zcr

def sgn(x):
    if x >= 0:
        return 1
    return -1

# 端点检测
def endPointCheck(frames, en, zcr):

    fn, wlen = frames.shape

    enHighThr = np.average(en) / 4 # 高能量阈值
    enLowThr = enHighThr / 4 # 低能量阈值
    zcrThr = np.average(zcr) - 0.2 * np.average(zcr[:5]) # 过零率阈值

    new = []
    labels = np.zeros(fn) # 值为status
    status = 0  # 0:无声段, 1:清音段, 2:语音段

    for i in range(fn):
        if en[i] >= enHighThr:
            k = j = i
            # 语音部分
            while (en[j] >= enLowThr and j < fn):
                labels[j] = 2
                j += 1
            while (en[k] >= enLowThr and k >= 0):
                labels[k] = 2
                k -= 1
            # 清音
            while (j < fn):
                if zcr[j] >= zcrThr:
                    labels[j] = 1
                j += 1
            while (zcr[k] >= zcrThr and k >= 0):
                k -= 1
                labels[k] = 1
    # print(labels)
    for i in range(fn):
        if labels[i] != 0:
            new.append(frames[i])
    new = np.array(new)

    return new

# 写入文件
def outputWaveData(frames, i, nchannels, sampwidth, framerate):
    # wave_data = []
    frames = frames.flatten().astype('int16')
    # frames = np.frombuffer(frames, dtype=np.short)
    # for n in range(len(frames)):
    #     if (frames[n] > 0):
    #         wave_data.append(frames[n])
    wave_data = np.array(frames)
    # wave_data.dtype = 'int32'
    print(wave_data, len(wave_data))
    f = wave.open("新语料\\" + str(i + 1) + ".pcm", "wb")
    # f.setnchannels(nchannels)
    # f.setsampwidth(sampwidth)
    # f.setframerate(framerate)
    f.setparams((nchannels, sampwidth, framerate, len(wave_data), 'NONE', 'HIT-1190201215'))
    f.writeframes(wave_data)
    f.close()


for i in range(10):
    file_path = "语料\\" + str(i+1) + ".wav"
    # print(file_path)
    wave_data, time, nchannels, sampwidth, framerate = getWaveData(file_path) # 从文件读取语音数据，返回采样点和采样点时间
    print('******************************')
    print(wave_data, len(wave_data))
    frames = findSegment(wave_data) # 语音分帧
    en = compute_en(frames, i) # 每帧能量
    zcr = compute_zcr(frames, i) # 每帧过零率
    new = endPointCheck(frames, en, zcr) # 端点检测
    outputWaveData(new, i, nchannels, sampwidth, framerate) # 写入文件




