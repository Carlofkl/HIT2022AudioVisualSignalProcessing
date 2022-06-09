

import numpy as np
import cv2
import math

# def CannyThreshold(lowThreshold = 100):
#     detected_edges = cv.GaussianBlur(gray,(3,3),0)
#     detected_edges = cv.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
#     dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
#     cv2.imshow('canny demo',dst)
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


img = cv2.imread('1.jpg', 0) # 灰度图
median = img.copy()
hist = img.copy()
dist = img.copy()
canny_img = img.copy()

# median = MedianFiltering(median)
# hist = histeq(hist)

# kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
# dst = cv2.filter2D(median, -1, kernel=kernel)


canny_img = cv2.Canny(canny_img, 80, 20)


cv2.namedWindow('canny')
cv2.imshow('canny', canny_img)
cv2.waitKey()
cv2.destroyAllWindows()