'''
实验目标
1. 掌握图像处理中读取、显示、保存。
2. 掌握图像处理中空域的增强算子。
3. 掌握图像直方图概念，实现图像的直方图均衡化。
实验内容
1. 实现图像的读取、显示、保存操作。（5分）
2. 实现图像的空域增强算子（中值滤波和均值滤波算法），显示并保存结果图像。（20分）
3. 实现图像的直方图均衡化，显示并保存结果图像。（25分）
实验要求
1. 本实验中仅图像的读取、显示、保存操作可以调用库函数，其他涉及到的图像算法均需自己写。
2. 编程语言为python3，要求代码格式规范，注释合理得当。
3. 建立自己的作业项目，代码中的文件地址需要是项目文件内的相对地址。
4. 需要在理解算法（算子）的内部原理的基础上进行编码，代码中要体现自己对算法（算子）的理解。
5. 寻找针对性的强的图像进行处理，方便结果展示。
'''

# import cv2.cv2 as cv
import cv2
import numpy as np
import math

#中值滤波算法
def MedianFiltering(img):
    h = img.shape[0]
    w = img.shape[1]
    medianfilter = []
    for i in range(h-2):
        for j in range(w-2):
            for ii in range(i, i+3):
                for jj in range(j, j+3):
                    medianfilter.append(img[ii, jj])
            MedNum = np.median(medianfilter)
            img[i+1, j+1] = MedNum
            medianfilter.clear()
    return img

#中值滤波快速算法
def medfilter(img, r=2):
    rows = img.shape[0]
    cols = img.shape[1]
    L = 2 * r + 1
    num = L * L
    cidx = num / 2 + 0.5
    out = np.zeros([rows, cols])
    for i in range(r, rows - r):
        hist = [0] * 256 # 计算第一列
        tmp = img[i - r:i + r + 1, 0:L].flatten()  # 获取直方图
        for n in range(0, num):
            hist[tmp[n]] += 1
        hist[tmp[n]] += 1
        histsum = 0         # 累积直方图
        for k in range(0, 256):
            histsum += hist[k]
            if histsum >= cidx:
                out[i, r] = k
                break
        # 后续计算
        for j in range(1 + r, cols - r):
            for m in range(-r, r + 1):
                tmp = img[i + m, j - 1 - r]
                hist[tmp] -= 1
                tmp = img[i + m, j + r]
                hist[tmp] += 1
            histsum = 0
            for k in range(0, 256):
                histsum += hist[k]
                if histsum >= cidx:
                    out[i, j] = k
                    break
    return out


#均值滤波算法
def MeanFiltering(img):
    h = img.shape[0]
    w = img.shape[1]
    medianfilter = []
    for i in range(h-2):
        for j in range(w-2):
            for ii in range(i, i+3):
                for jj in range(j, j+3):
                    medianfilter.append(img[ii, jj])
            MedNum = np.mean(medianfilter)
            img[i+1, j+1] = MedNum
            medianfilter.clear()
    return img


#图像直方图均衡化
def histeq(img):
    # 计算image的尺寸，可以得到像素总数
    h = img.shape[0]
    w = img.shape[1]
    # print(h * w)
    # print(w)

    #计算每个灰度的像素个数，hist[i]存储灰度值为i的像素数量，遍历整个image
    hist = np.zeros(256)
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1

    #（灰度分布直方图），计算每个灰度像素数占总数的比值
    hist = hist / (h * w)

    #计算累计分布函数，例如hist[i]存储着原先hist[0]到hist[i]所有像素之和
    for i in range(1, 256):
        hist[i] = hist[i] + hist[i-1]

    #计算映射后的g,并向下取整，公式为(gmax-gmin)xC(f)+gmin+0.5，并向下取整
    hist = 255 * hist + 0.5
    for i in range(256):
        hist[i] = math.floor(hist[i])

    #映射到原图像，得到原先的灰度应该映射到哪个新的灰度
    for i in range(h):
        for j in range(w):
            img[i, j] = hist[img[i, j]]

    return img


image1 = cv2.imread('Fig0504(i)(salt-pepper-noise).tif', 0) # 图像读取,椒盐噪声
image2 = cv2.imread('Fig0316(2)(2nd_from_top).tif', 0) # 灰度直方图，图像读取
image_1 = image1.copy()
image_2 = image1.copy()
image_3 = image2.copy()

#图像处理
# image_1 = MedianFiltering(image_1) #中值滤波
image_1 = medfilter(image_1) #中值滤波快速算法
image_2 = MeanFiltering(image_2) #均值滤波
imgs1 = np.hstack([image1, image_1, image_2])

image_3 = histeq(image_3)
imgs2 = np.hstack([image2, image_3])

#增强算子，图像显示
cv2.namedWindow('mutil_pic1')
cv2.imshow('mutil_pic1', imgs1)
cv2.waitKey()
cv2.destroyAllWindows()

#直方图均衡化，图像显示
cv2.namedWindow('mutil_pic2')
cv2.imshow('mutil_pic2', imgs2)
cv2.waitKey()
cv2.destroyAllWindows()

#图像保存
cv2.imwrite('D:\codes4\CV\lab1\mdeian_filter.jpg', image_1) #中值滤波，图像保存
cv2.imwrite('D:\codes4\CV\lab1\mean_filter.jpg', image_2) #均值滤波，图像保存
cv2.imwrite('D:\codes4\CV\lab1\histeq.jpg', image_3) #直方图均值化，图像保存