import cv2

# 用opencv 复现 threshold(GrayImage, Region, 180, 255)

img = cv2.imread("123.png", cv2.IMREAD_GRAYSCALE)
gray = img  # 直接使用灰度图

cv2.imshow('src', img)

# 第一次阈值分割：低阈值
ret1, thres_low = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
cv2.imshow('thres_low', thres_low)

# 第二次阈值分割：高阈值
ret2, thres_high = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
cv2.imshow('thres_high', thres_high)

# 两次阈值图像相减，保留中间灰度段的像素区域
threshold = thres_low - thres_high

num_labels, labels = cv2.connectedComponents(threshold)

# 遍历连通区域
for i in range(1, num_labels):  # label=0 是背景
    region_mask = (labels == i).astype('uint8') * 255
    cv2.imshow(f"Region {i}", region_mask)