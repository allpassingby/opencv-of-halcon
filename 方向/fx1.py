import cv2
import numpy as np

# 读取图像
image = cv2.imread('E:/ruanjian/HALCON-25.05-Progress/cs/biaoqian/right.png')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 显示原始灰度图像
cv2.imshow("Gray Image", gray)

# 使用阈值进行二值化处理
_, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

# 使用开运算去除小的噪点
kernel = np.ones((5, 5), np.uint8)
opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# 查找连通区域
contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 过滤掉面积不在指定范围的轮廓
min_area = 9000  # 最小面积
max_area = 12000  # 最大面积
filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

# 确保筛选后的轮廓不为空
if filtered_contours:
    # 找到最大的轮廓（应该是箭头）
    arrow_contour = max(filtered_contours, key=cv2.contourArea)

    # 计算轮廓的最小外接矩形
    rect = cv2.minAreaRect(arrow_contour)
    angle = rect[2]

    # 调整角度，使其在[-45, 45]范围内
    if angle < -45:
        angle = 90 + angle

    # 计算箭头的质心
    moments = cv2.moments(arrow_contour)
    cx = int(moments['m10'] / moments['m00'])  # 计算质心X坐标
    cy = int(moments['m01'] / moments['m00'])  # 计算质心Y坐标

    # 获取最小外接矩形的中心点
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    rect_center = tuple(np.mean(box, axis=0))

    # 通过质心与矩形中心的相对位置判断箭头方向
    if cx > rect_center[0]:
        direction = "Right"
    else:
        direction = "Left"

    print(f"箭头指向: {direction}")
    cv2.putText(image, f"Arrow points {direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 绘制最小外接矩形
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

# 显示结果图像
cv2.imshow("Detected Arrow", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
