'''
1. 设计命令词识别任务
	设想一个任务，如智能家居、或车辆控制等
	确定词表，要求词表中不少于10个词
	录制语料。采集特定人（自己）语料，每个词不少于五遍。取其中一遍为模板，其它四遍用来测试。可以用采集工具(如cooledit)或编程实现语音采集。
	检查语料。通过听辩检查保证语料质量。
	去除静音。可以用端点检测算法实现，也可以手工实现

2. 特征提取
	每帧提取39维MFCC特征，帧长25ms，帧移10ms
	可以采用HTK工具包中的hcopy命令实现（要求语料是WAV格式）
	hcopy -A -D -T 1 -C tr_wav.cfg -S .\data\list.scp

3. 识别测试
	N个模板，M个待测命令词语料，进行N*M次DTW计算，得到N*M个DTW距离
	    分别载入模板和待测语料的MFCC特征矢量序列
		计算两个特征序列的 DTW 距离

4. 计算测试结果
	每个测试语料都有一个类别标签 l_i
	每个测试语料都有一个识别结果 r_i
	r_i = maxD_{ij}
	其中， D_{ij}为第i个测试语料和第j个模板间的DTW距离（规整后）
	若 r_i = l_i 表示识别结果正确
	计算正确率=识别结果正确的语料数/总测试语料数

5. 扩展尝试
	开集扩展：采集一批集外命令词，重新计算正确率？
	实用扩展：将经过实验验证的算法，转化为能实时采集，在线检测的命令词识别系统？（这里我做的就是这个扩展）
	算法扩展：尝试基于HMM的识别算法？
'''

import os
import math
import wave
import numpy as np
from struct import unpack
from itertools import product
import matplotlib.pylab as plt
import EndPointDetection

def getMFCC():
    MFCC = []
    for i in range(5):
        MFCC_row = []
        for j in range(10):
            f = open("语料\\"+str(i+1)+"_"+str(j+1)+".mfc", "rb")
            nframes = unpack(">i", f.read(4))[0] # 帧数
            frate = unpack(">i", f.read(4))[0]  # 帧频 100ns
            nbytes = unpack(">h", f.read(2))[0]  # 特征的字节数
            feakind = unpack(">h", f.read(2))[0] # 9 is user
            # print(feakind)
            data = []
            ndim = nbytes / 4 # 特征的维数，每个特征4B
            # print(nbytes, ndim, nframes)
            for k in range(nframes):
                data_frame = []
                for l in range(int(ndim)):
                    data_frame.append(unpack(">f", f.read(4))[0])
                data.append(data_frame)
            f.close()
            MFCC_row.append(data)
        MFCC.append(MFCC_row)
    return MFCC

def getMFCC_model_test(MFCC, type):
    model = []
    test = []
    if type == 'DTW':
        for i in range(5):
            test_row = []
            model.append(MFCC[i][0])
            for j in range(1, 10):
                test_row.append(MFCC[i][j])
            test.append(test_row)
        return model, test

    elif type == 'HMM':
        for i in range(5):
            model_row = []
            test_row = []
            for j in range(7):
                model_row.append(MFCC[i][j])
            for k in range(7, 10):
                test_row.append((MFCC[i][k]))
            model.append(model_row)
            test.append(test_row)
        return model, test

    else:
        return -1



def voiceRecognition(model, test, type):
    row = len(test)
    col = len(test[0])
    cnt = 0  # 正确的语料数
    flag_r = np.zeros((5, 9))  # 待测试集的标签
    if type == "DTW":
        for i in range(row):
            flag_l = i + 1
            for j in range(col):
                print("正在处理第"+str(i*col+1+j)+"个音频")
                flag_r[i][j] = 1
                min_dis = DTW(test[i][j], model[0])
                for k in range(len(model)):
                    dis = DTW(test[i][j], model[k])
                    if dis < min_dis:
                        flag_r[i][j] = k + 1
                        min_dis = dis
                if flag_r[i][j] == flag_l:
                    cnt += 1
        print("待测试集标签为：")
        print(flag_r)
        print("正确率为：")
        print(cnt / (row * col))


    elif type == "HMM":
        return -1
    else:
        print('ERROR')

# 自测函数，看DTW有没有编错
def voiceRecognition1(model, test, type):
    cnt = 0  # 正确的语料数
    flag_r = np.zeros((5, 1))  # 待测试集的标签
    if type == "DTW":
        for i in range(5):
            flag_l = i + 1
            flag_r[i] = 1
            min_dis = DTW(test[i], model[0])
            for k in range(5):
                dis = DTW(test[i], model[k])
                if dis < min_dis:
                    flag_r[i] = k + 1
                    min_dis = dis
            if flag_r[i] == flag_l:
                cnt += 1
        print("待测试集标签为：")
        print(flag_r)
        print("正确率为：")
        print(cnt / 5)


    elif type == "HMM":
        return -1
    else:
        print('ERROR')

def DTW(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    row = len(x1)
    col = len(x2)
    ndim = len(x1[0])
    D = np.zeros((row, col))
    # 初始化（i，j）的代价
    for i in range(row):
        for j in range(col):
            D[i][j] = 0
            D[i][j] = np.linalg.norm(x1[i] - x2[j]) # 欧式距离
            # for k in range(ndim):
            #     D[i][j] += abs(x1[i][k] - x2[j][k]) # 每一个差值的绝对值，最后求和
    for i in range(1, row): # 计算每一列代价路径的累计距离
        D[i][0] += D[i-1][0]
    for j in range(1, col): # 计算每一行代价路径的累计距离
        D[0][j] += D[0][j-1]
    for i in range(1, row): # 开始搜索最短代价路径
        for j in range(1, col):
            D[i][j] = min(D[i-1][j]+D[i][j], D[i-1][j-1]+2*D[i][j], D[i][j-1]+D[i][j])
    return D[row-1][col-1] / (row + col -2) # 结果用系数和归正
    # return D[row-1][col-1]



type = 'DTW'
# tpye = 'HMM'
MFCC = getMFCC()
model, test = getMFCC_model_test(MFCC, type)
voiceRecognition(model, test, type)
# voiceRecognition1(model, model, type)

