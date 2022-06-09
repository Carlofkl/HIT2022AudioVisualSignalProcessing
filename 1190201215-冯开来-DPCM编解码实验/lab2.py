'''
第一步：从wav文件中读入原始数据
第二步：与解码后的x'(n-1)计算差值
第三步：对从d(n)进行重新量化为c(n)，要求8bit，一个采样一个BYTE
    直接量化或量化因子法
第四步：解码
第五步：对从d(n)进行重新量化，要求4bit，两个采样一个BYTE
    直接量化或量化因子法
第六步：解码计算 x'(n-1)
    量化因子法 对数变换法
第七步：封装c(n)
第八步：保存编码到文件（后缀用”.dpc”）

改进思路
计算信噪比


'''

import os
import math
import wave
import numpy as np
import matplotlib.pylab as plt

# 读取文件
def readWaveData(file_path):
    f = wave.open(file_path, "rb")
    params = f.getparams()
    # 声道数，量化位数，采样频率，采样点数
    nchannels, sampwidth, framerate, nframes = params[:4]
    # 得到每个采样点的值
    str_data = f.readframes(nframes)
    f.close()
    # 转成short类型
    wave_data = np.frombuffer(str_data, dtype=np.short)
    # 通过采样点数和取样频率计算每个取样的时间
    time = np.arange(0, nframes) / framerate
    return wave_data, time, nchannels, sampwidth, framerate



def DPCM(data, method, bit):
    a = 110 # 量化因子 8bit
    b = 1260 # 量化因子 4bit
    q = 1 # 自己优化方法中的量化因子
    p = 3 # 自己优化方法中取对数的底数
    n = len(data)
    x = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)
    x[0] = data[0]
    d[0] = data[0]

    # method: 0-直接量化法，1-量化因子法，2-自己优化的方法，结合了对数变换和量化因子法
    # bit: 4-4bit量化，8-8bit量化

    # 8-bit直接量化
    if method == 0 and bit == 8: # 8-bit直接量化
        # 对第一位进行编码
        if d[0] > 127:
            c[0] = 255
        elif d[0] < -128:
            c[0] = 0
        else:
            c[0] = d[0]
        # 对接下来的每一位差值进行编码
        for i in range(1, n):
            d[i] = data[i] - x[i-1]
            if d[i] > 127:
                c[i] = 127
            elif d[i] < -128:
                c[i] = -128
            else:
                c[i] = d[i]
            c[i] += 128
            # 在量化的过程中嵌入了解码的过程
            x[i] = x[i-1] + c[i]-128 # 解码过程
        print(cal_snr(wave_data, x)) # 计算信噪比

        # 将预测的音频写入文件，即解码的文件
        f = wave.open("DPCM\\2_" + str(bit) +"bit.pcm", "wb")
        f.setparams((1, 2, 16000, len(x), 'NONE', 'HIT-1190201215'))
        f.writeframes(x.astype(np.short))
        f.close()
        # 将cn写入文件，即编码文件，c[0]是16-bit，其余皆是一个采样点一个byte
        g = open("DPCM\\2_" + str(bit) + "bit.dpc", "wb")
        g.write(np.short(c[0]))
        for i in range(1, len(c)):
            g.write(np.int8(c[i]))
        g.close()

        return x, c

    # 8-bit量化因子
    elif method == 1 and bit == 8:
        # 对第一位进行编码
        if d[0] > 127 * a:
            c[0] = 127
        elif d[0] < -128 * a:
            c[0] = -128
        else:
            for j in range(-128, 128):
                if (j-1) * a < d[0] <= j * a:
                    c[0] = j
                    break
        # 对剩下的差值进行编码
        for i in range(1, n):
            d[i] = data[i] - x[i - 1]
            if d[i] > 127 * a:
                c[i] = 127
            elif d[i] < -128 * a:
                c[i] = -128
            else:
                for j in range(-128, 128):
                    if (j - 1) * a < d[i] <= j * a:
                        c[i] = j
                        break
            c[i] += 128 # 每个c[i]加上128也是编码过程，我的认为是为了’居中‘
            x[i] = x[i - 1] + (c[i] - 128) * a # 解码过程

        print(cal_snr(wave_data, x))

        f = wave.open("DPCM\\2_" + str(bit) +"bit.pcm", "wb")
        f.setparams((1, 2, 16000, len(x), 'NONE', 'HIT-1190201215'))
        f.writeframes(x.astype(np.short))
        f.close()

        c_dpc = np.zeros((c.shape[0]))
        g = open("DPCM\\2_" + str(bit) + "bit.dpc", "wb")
        g.write(np.short(c[0]))
        for i in range(1, len(c)):
            g.write(np.int8(c[i]))
        g.close()

        return x, c

    # 4-bit直接量化
    elif method == 0 and bit == 4:
        # 对一个差值进行量化
        if d[0] > 7:
            c[0] = 7
        elif d[0] < -8:
            c[0] = -8
        else:
            c[0] = d[0]
        for i in range(1, n): # 对剩下的差值进行量化
            d[i] = data[i] - x[i-1]
            if d[i] > 7:
                c[i] = 7
            elif d[i] < -8:
                c[i] = -8
            else:
                c[i] = d[i]
            c[i] += c[i] + 8
            x[i] = x[i-1] + (c[i] - 8) # 解码过程
        print(cal_snr(wave_data, x))
        # 将预测的数据（解码数据）写入文件
        f = wave.open("DPCM\\2_" + str(bit) + "bit.pcm", "wb")
        f.setparams((1, 2, 16000, len(x), 'NONE', 'HIT-1190201215'))
        f.writeframes(x.astype(np.short))
        f.close()
        #将c(n)写入文件
        lengh = int(np.ceil((c.shape[0]+1) / 2)) # 计算两个采集点为一个byte的数组长度
        c_dpc = np.zeros(lengh) # 初始化新数组，为即将写入文件的编码
        g = open("DPCM\\2_" + str(bit) + "bit.dpc", "wb")
        c_dpc[0] = c[0]
        g.write(np.short(c_dpc[0])) # 首位仍然采用16bit写入文件
        for i in range(1, len(c)-1, 2): # 步长为2，为放置溢出，终止条件为原c(n)-1
            n = int(np.ceil(i/2))
            c_dpc[n] = c[i]*16 + c[i+1]
            g.write(np.int8(n))
        if len(c) % 2 == 0: # 当原数组是2的整数倍也就是除去首位后无法被2整除，则最后一位单独处理
            n = lengh -1
            c_dpc[n] = c[len(c)-1]*16
            g.write(np.int8(c_dpc[n]))
        g.close()

        return x, c_dpc

    # 4-bit量化因子
    elif method == 1 and bit == 4:
        if d[0] > 7 * b:
            c[0] = 7
        elif d[0] < -8 * b:
            c[0] = -8
        else:
            for j in range(-8, 8):
                if (j-1) * b < d[0] <= j * b:
                    c[0] = j
                    break
        for i in range(1, n):
            d[i] = data[i] - x[i-1]
            if d[i] > 7 * b:
                c[i] = 7
            elif d[i] < -8 * b:
                c[i] = -8
            else:
                for j in range(-8, 8):
                    if (j - 1) * b < d[i] <= j * b:
                        c[i] = j
                        break
            c[i] += 8
            x[i] = x[i-1] + (c[i] - 8) * b # 解码过程
        print(cal_snr(wave_data, x))

        f = wave.open("DPCM\\2_" + str(bit) + "bit.pcm", "wb")
        f.setparams((1, 2, 16000, len(x), 'NONE', 'HIT-1190201215'))
        f.writeframes(x.astype(np.short))
        f.close()

        lengh = int(np.ceil((c.shape[0]+1) / 2))
        c_dpc = np.zeros(lengh)
        g = open("DPCM\\2_" + str(bit) + "bit.dpc", "wb")
        c_dpc[0] = c[0]
        g.write(np.short(c_dpc[0]))
        for i in range(1, len(c)-1, 2):
            n = int(np.ceil(i/2))
            c_dpc[n] = c[i]*16 + c[i+1]
            g.write(np.int8(n))
        if len(c) % 2 == 0:
            n = lengh -1
            c_dpc[n] = c[len(c)-1]*16
            g.write(np.int8(c_dpc[n]))
        g.close()

        return x, c_dpc

    # 4-bit 自己优化的方法，结合了对数变换和量化因子法
    elif method == 2 and bit == 4:
        c[0] = np.log(abs(d[0])) / np.log(p)
        if c[0] < -7*q:
            c[0] = -7
        elif c[0] > 8*q:
            c[0] = 8
        else:
            for j in range(-7, 7):
                if (j-1)*q < c[0] <= j*q:
                    c[0] = j
                    break
        for i in range(1, n):
            d[i] = data[i] - x[i-1]
            if d[i] == 0:
                d[i] = 1
            c[i] = np.log(abs(d[i])) # 先对dn取对数，然后对对数进行量化因子法
            if c[i] < -7 * q:
                c[i] = -7
            elif c[i] > 8 * q:
                c[i] = 8
            else:
                for j in range(-7, 7):
                    if (j - 1) * q < c[i] <= j * q:
                        c[i] = j
                        break
            c[i] += 8
            # 在解码的过程中要注意dn的正负
            if d[i] > 0:
                x[i] = x[i-1] + pow(p, (c[i]-8) * q)
            else:
                x[i] = x[i-1] - pow(p, (c[i]-8) * q)

        print(cal_snr(wave_data, x))

        f = wave.open("DPCM\\2_" + str(bit) + "bit.pcm", "wb")
        f.setparams((1, 2, 16000, len(x), 'NONE', 'HIT-1190201215'))
        f.writeframes(x.astype(np.short))
        f.close()

        lengh = int(np.ceil((c.shape[0]+1) / 2))
        c_dpc = np.zeros(lengh)
        g = open("DPCM\\2_" + str(bit) + "bit.dpc", "wb")
        c_dpc[0] = c[0]
        g.write(np.short(c_dpc[0]))
        for i in range(1, len(c)-1, 2):
            n = int(np.ceil(i/2))
            c_dpc[n] = c[i]*16 + c[i+1]
            g.write(np.int8(n))
        if len(c) % 2 == 0:
            n = lengh -1
            c_dpc[n] = c[len(c)-1]*16
            g.write(np.int8(c_dpc[n]))
        g.close()

        return x, c_dpc


    else:
        print('error!')
        return -1, -1

def sgn(x):
    if x >= 0:
        return 1
    return -1

def cal_snr(x, X):
    lengh = len(x)
    s1 = np.dot(x/lengh, x.T/lengh) # 除以lengh归一化处理，防止溢出
    s2 = np.dot((x-X)/lengh, (x-X).T/lengh)
    return 10 * np.log10(s1/s2)


file_path = "语料\\2.wav"
# 从文件读取语音数据，返回采样点和采样点时间
wave_data, time, nchannels, sampwidth, framerate = readWaveData(file_path)
# print(wave_data, len(wave_data))
# method: 0直接量化法，1量化因子法，2自己优化的方法，结合了对数变换和量化因子法
# bit: 4-4bit量化，8-8bit量化
print("信噪比如下：")
x, c = DPCM(wave_data, method=1, bit=4)
# print(x, c, len(x))
