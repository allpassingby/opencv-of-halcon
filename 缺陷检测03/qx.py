from skimage.feature import graycomatrix
from skimage.measure import regionprops
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 初始化
def display_image(image, window_name='Image'):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()  # 销毁显示的窗口


# 计算背景照明（简化版本）
def estimate_background_illumination(image):
    # 假设背景照明是低频部分（可以通过高斯滤波来估计）
    background = cv2.GaussianBlur(image, (21, 21), 0)
    return background


# 中值滤波
def median_filter(image, kernel_size=9):
    return cv2.medianBlur(image, kernel_size)


# 分水岭算法
def watershed_segmentation(image):
    # 归一化并找到阈值
    ret, threshed = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)

    # 需要确保图像是3通道的彩色图像
    image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # 转换为彩色图像

    # 进行分水岭分割
    dist_transform = cv2.distanceTransform(threshed, cv2.DIST_L2, 5)
    _, markers = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    markers = np.uint8(markers)

    # 确保 markers 是 32 位整数类型
    markers = markers.astype(np.int32)  # 转换为32位整数

    cv2.watershed(image_colored, markers)  # 使用彩色图像和32位整数标记图像

    return markers


# 计算图像的共生矩阵特征（能量、相关性、同质性、对比度）
def cooc_feature_image(markers, image):
    # 使用skimage的regionprops来提取区域信息
    regions = regionprops(markers)

    energy, contrast, homogeneity, correlation = [], [], [], []

    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox  # 获取区域的边界框

        # 提取子图像
        sub_image = image[min_row:max_row, min_col:max_col]

        # 计算灰度共生矩阵
        glcm = graycomatrix(sub_image, distances=[1], angles=[0], symmetric=True, normed=True)

        # 计算共生矩阵特征
        energy_val = np.sum(glcm ** 2)
        contrast_val = np.sum(np.square(glcm - np.mean(glcm)))
        homogeneity_val = np.sum(glcm / (1 + np.abs(glcm - np.mean(glcm))))
        correlation_val = np.corrcoef(glcm.flatten(), glcm.flatten())[0, 1]

        energy.append(energy_val)
        contrast.append(contrast_val)
        homogeneity.append(homogeneity_val)
        correlation.append(correlation_val)

    return energy, contrast, homogeneity, correlation


# 主要处理过程
def process_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 估计背景照明
    background = estimate_background_illumination(image_gray)

    # 图像减去背景照明
    image_sub = cv2.subtract(image_gray, background)

    # 中值滤波
    image_median = median_filter(image_sub)

    # 分水岭算法
    markers = watershed_segmentation(image_median)

    # 对分水岭算法的结果进行共生矩阵特征计算
    energy, contrast, homogeneity, correlation = cooc_feature_image(markers, image_median)

    # 调整能量阈值来选定缺陷
    defects_mask = np.where(np.array(energy) <= 0.05, 255, 0).astype(np.uint8)

    # 显示结果
    result_image = image.copy()
    contours, _ = cv2.findContours(defects_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 10:
            # 在缺陷区域绘制红色矩形框
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色矩形框

            # 在区域中心画圆
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            cv2.circle(result_image, (cx, cy), 10, (0, 0, 255), 2)  # 红色圆

    # 调用显示函数，显示最终结果
    display_image(result_image)
    print(f'{len(contours)} mura defects detected')


# 主程序
if __name__ == '__main__':
    process_image('mura_defects_texture_02.png')  # 替换为实际图像路径
