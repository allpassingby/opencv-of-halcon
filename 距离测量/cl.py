import cv2
import numpy as np
import math

# 读取图像
image = cv2.imread('161250208.bmp', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("图像加载失败，请检查路径或图像文件。")
else:
    # 创建测量线段的两端点
    pt1 = (int(1066.04), int(658.477))
    pt2 = (int(1278.6), int(655.964))

    # 计算中心、角度和长度
    center_x = (pt1[0] + pt2[0]) / 2
    center_y = (pt1[1] + pt2[1]) / 2
    dx = pt2[0] - pt1[0]
    dy = pt1[1] - pt2[1]
    length = math.hypot(dx, dy)
    angle = math.atan2(dy, dx) * 180 / np.pi  # 注意 OpenCV 用的是角度

    # 定义 ROI 宽度
    roi_width = 29 * 2  # Halcon 是 len2 的两倍

    # 计算旋转矩形
    rect = ((center_x, center_y), (length, roi_width), angle)

    # 提取 ROI 区域
    box = cv2.boxPoints(rect).astype(np.intp)
    mask = np.zeros_like(image, dtype=np.uint8)  # 强制指定类型为 np.uint8
    cv2.drawContours(mask, [box], 0, 255, -1)  # 绘制旋转矩形的掩膜
    roi = cv2.bitwise_and(image, mask)

    # 对 ROI 旋转矫正（warpAffine）
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    roi_rotated = cv2.warpAffine(mask, M, (image.shape[1], image.shape[0]))
    roi_cut = cv2.bitwise_and(rotated, roi_rotated)

    # 裁剪中间区域（即水平方向的灰度曲线）
    x, y, w, h = cv2.boundingRect(cv2.findNonZero(roi_rotated))
    strip = roi_cut[y:y+h, x:x+w]

    # 沿水平方向（列）进行灰度投影（即边缘提取）
    intensity_profile = np.mean(strip, axis=0)

    # 边缘检测：可用一阶微分（梯度）找到最大变化点
    gradient = np.gradient(intensity_profile)

    # 找两个主边缘（最大正负变化）
    peak1 = np.argmax(gradient)
    peak2 = np.argmin(gradient)

    # 计算距离（可转成物理距离）
    pixel_distance = abs(peak2 - peak1)

    # 显示图像与测量值
    cv2.imshow("ROI Strip", strip)
    print("测量距离（像素）:", pixel_distance)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
