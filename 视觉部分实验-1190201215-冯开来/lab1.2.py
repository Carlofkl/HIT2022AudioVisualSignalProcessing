'''
实验内容
1. 选择合适的图像处理算法找到图1（源文件见附件）中的橘子和枣子的数量。（25分）
2. 在1的基础上确定每个水果的外边界，并使用边界线或者mask将属于水果的像素点标注出来。（25分）
图1 橘子们和枣子们
实验要求
1. 提示：只需要根据水果的大小差别区分出水果的不同种类即可。
2. 编程语言为python3，库函数可自行选用，要求代码格式规范，注释合理得当。
3. 建立自己的作业项目，代码中的文件地址需要是项目文件内的相对地址。
4. 需要在理解所选函数的内部原理的基础上进行编码，报告中需要体现自己对函数的详细理解。
5. 鼓励尝试多种方式完成作业。
'''
import cv2
import numpy as np
import math
import matplot as plt

# 似乎没有用到的canny算子
def CannyThreshold(lowThreshold = 100):
    detected_edges = cv.GaussianBlur(gray,(3,3),0)
    detected_edges = cv.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('canny demo',dst)

# 二值化
def linear_threshold(img, x1 = 95, x2 = 8):
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            tmp = img[i, j]
            if tmp < x2:
                img[i, j] = 255
            elif tmp >= x2 and tmp < x1:
                img[i, j] = 0
            else:
                img[i ,j] = 255
    return img

# 过滤一些小的轮廓，并根据面积大小判断橘子或枣子
def Filter(contours):
    Sum = 0 # 枣子
    sum = 0 # 橘子
    a = []
    for i, area in enumerate(contours):
        if cv2.contourArea(area) > 8000 and cv2.contourArea(area)<12000:
            a.append(contours[i])
            Sum += 1
        elif cv2.contourArea(area) > 12000 and cv2.contourArea(area)<100000:
            a.append(contours[i])
            sum += 1
        else:
            continue
    return Sum, sum, a


# 主函数
image = cv2.imread('tangerine&dates.jpg', 1) # 原图
img = cv2.imread('tangerine&dates.jpg', 0) # 灰度图
Sum = 0
sum = 0

# # canny算子
# canny_img = img.copy()
# canny_img = cv2.Canny(canny_img,95,8)
#
# # 找轮廓
# contours,hierarchv = cv2.findContours(canny_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

# color = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
# Sum, sum, contours = Filter(contours)

# # 灰度直方图
# hist_mask = cv2.calcHist([img], [0], None, [256], [0, 256])

'''
二值化之后膨胀腐蚀
'''

# 二值化
threshold_image = img.copy()
threshold_image = linear_threshold(threshold_image)

# 膨胀腐蚀（似乎没有必要）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
dstImg = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel)

# 找到轮廓
contours,hierarchv = cv2.findContours(dstImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# color = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
contours1 = contours.copy()
Sum, sum, contours1 = Filter(contours1)

'''
显示
'''

# 在原图上画轮廓
image1 = image.copy()
image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
image1 = cv2.drawContours(image1, contours1, -1, (100, 0, 255), 3)


# 原图
cv2.namedWindow('gray')
cv2.imshow('gray', img)

# # Canny算子
# cv2.namedWindow('Canny')
# cv2.imshow('Canny', img)

# 二值化
cv2.namedWindow('threshold')
cv2.imshow('threshold', threshold_image)

# 膨胀腐蚀
cv2.namedWindow('open')
cv2.imshow('open', dstImg)

# 轮廓
cv2.namedWindow('contours')
cv2.imshow('contours', image)

# 结果
cv2.namedWindow('result')
cv2.imshow('result', image1)

# 展示图片
cv2.waitKey()
cv2.destroyAllWindows()
print("枣子的数量："+str(Sum))
print('橘子的数量：'+str(sum))





# img1 = cv2.Canny(img,80,150)
# h = cv2.findContours(binaryImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# #提取轮廓
# contours = h[0]
# #打印返回值，这是一个元组
# print(type(h))
# #打印轮廓类型，这是个列表
# print(type(h[1]))
# #查看轮廓数量
# print (len(contours))
# temp = np.ones(binaryImg.shape,np.uint8)*255
#画出轮廓：temp是白色幕布，contours是轮廓，-1表示全画，然后是颜色，厚度
# cv2.drawContours(img,contours,-1,(255,0,0),3)
#
# cv2.imshow("contours",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


