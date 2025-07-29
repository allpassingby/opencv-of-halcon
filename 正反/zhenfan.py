import cv2
import os
import numpy as np

# 设置图像文件夹路径
image_folder = 'E:/ruanjian/HALCON-25.05-Progress/cs/zhengfan'

# 获取所有图片文件（支持的格式）
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.tif', '.tiff', '.gif', '.bmp', '.jpg', '.jpeg', '.jp2', '.png', '.pcx', '.pgm', '.ppm', '.pbm', '.xwd', '.ima', '.hobj'))]

# 遍历所有图像文件
for image_file in image_files:
    # 读取图像
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    # 定义矩形区域并裁剪
    row1, col1 = 645, 710
    row2, col2 = 1433, 1500
    rectangle_roi = image[row1:row2, col1:col2]

    # 转为灰度图像
    gray_image = cv2.cvtColor(rectangle_roi, cv2.COLOR_BGR2GRAY)

    # 计算灰度区域的平均亮度
    mean_brightness = np.mean(gray_image)

    # 判断正面或背面
    if mean_brightness > 114:
        direction = '正面'
    else:
        direction = '背面'

    # 输出判断结果
    print(f'图像：{image_path} 方向：{direction}')
    print(mean_brightness)

    # 在图像上添加方向文本
    cv2.putText(image, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 在原图像上绘制矩形框，显示裁剪区域
    cv2.rectangle(image, (col1, row1), (col2, row2), (0, 255, 0), 2)

    # 显示图像
    cv2.namedWindow(f'Processed Image - {image_path}', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(f'Processed Image - {image_path}', 800, 600)  # 设置窗口大小为 800x600
    cv2.imshow(f'Processed Image - {image_path}', gray_image)

    # 等待用户按键后关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
