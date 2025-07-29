import cv2
import numpy as np

# 指定单张图像路径
image_path = 'E:/ruanjian/HALCON-25.05-Progress/cs/zhengfan/15-19-02-3347-00.jpg'

# 读取图像
image = cv2.imread(image_path)

# 生成一个圆形ROI，坐标和半径
center = (991, 1067)  # 圆心坐标
radius = 265  # 半径
mask = np.zeros(image.shape[:2], dtype=np.uint8)  # 创建一个空白图像作为掩码
cv2.circle(mask, center, radius, 255, -1)  # 在掩码上画圆

# 裁剪出圆形区域
image_roi = cv2.bitwise_and(image, image, mask=mask)

# 获取图像大小
height, width = image.shape[:2]

# 定义矩形区域并裁剪
row1, col1 = 645, 710
row2, col2 = 1433, 1500
rectangle_roi = image[row1:row2, col1:col2]

# 转为灰度图像
gray_image = cv2.cvtColor(rectangle_roi, cv2.COLOR_BGR2GRAY)

# 计算平均亮度
mean_brightness = np.mean(gray_image)

# 判断正面或背面
if mean_brightness > 58:
    direction = '正面'
else:
    direction = '背面'

# 输出判断结果
print(f'图像：{image_path} 方向：{direction}')
#print('direction')

# 将图像调整为适合显示的大小，避免显示过大
resize_factor = 0.5  # 控制缩放比例
resized_image = cv2.resize(image, (int(width * resize_factor), int(height * resize_factor)))

# 在图像上添加方向文本
cv2.putText(resized_image, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# 设置显示窗口大小
cv2.namedWindow(f'Processed Image - {image_path}', cv2.WINDOW_NORMAL)
cv2.resizeWindow(f'Processed Image - {image_path}', 800, 600)  # 设置窗口大小为 800x600

# 显示图像和结果
#cv2.imshow(f'Processed Image - {image_path}', resized_image)

# 等待用户按键后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()


