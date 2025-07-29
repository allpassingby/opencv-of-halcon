import cv2
import numpy as np
import os
import time

# 1. 设置模型图片和道路图片文件路径
image_folder = os.getcwd()  # 当前目录
attention_sign_path = os.path.join(image_folder, 'attention_road_sign.jpg')  # 注意标志图像路径
dead_end_sign_path = os.path.join(image_folder, 'dead_end_road_sign.jpg')  # 死胡同标志图像路径

# 2. 读取标志图像（Attention 和 Dead End）
attention_sign = cv2.imread(attention_sign_path, cv2.IMREAD_COLOR)
dead_end_sign = cv2.imread(dead_end_sign_path, cv2.IMREAD_COLOR)

# 检查图像是否正确加载
if attention_sign is None or dead_end_sign is None:
    print("Error loading sign images. Please check the file paths.")
    exit()

# 3. 创建窗口以显示图像
cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

# 初始化匹配参数
scaleRMin = [0.5, 0.4]
scaleRMax = [0.8, 2.0]
scaleCMin = [1.0, 1.0]
scaleCMax = [1.0, 1.0]
hFac = [47.0, 50.0]  # 根据实际标志尺寸设置缩放因子

# 4. 读取当前目录下所有的 PNG 图像
street_image_files = [f for f in os.listdir(image_folder) if
                      f.endswith('.jpg') and f not in ['attention_road_sign.png', 'dead_end_road_sign.png']]

# ORB 特征检测器
orb = cv2.ORB_create()

# 5. 遍历图像序列进行标志检测
for street_image_file in street_image_files:
    # 读取街道图像
    street_image_path = os.path.join(image_folder, street_image_file)
    street_image = cv2.imread(street_image_path, cv2.IMREAD_COLOR)

    if street_image is None:
        continue

    # 使用 ORB 特征检测
    kp_attention, des_attention = orb.detectAndCompute(attention_sign, None)
    kp_dead_end, des_dead_end = orb.detectAndCompute(dead_end_sign, None)
    kp_image, des_image = orb.detectAndCompute(street_image, None)

    # 使用 BFMatcher 匹配特征
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches_attention = bf.match(des_image, des_attention)
    matches_dead_end = bf.match(des_image, des_dead_end)

    # 根据匹配结果排序
    matches_attention = sorted(matches_attention, key = lambda x:x.distance)
    matches_dead_end = sorted(matches_dead_end, key = lambda x:x.distance)

    # 6. 显示匹配结果
    attention_start_time = time.time()
    result_attention = cv2.drawMatches(street_image, kp_image, attention_sign, kp_attention, matches_attention[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    dead_end_start_time = time.time()
    result_dead_end = cv2.drawMatches(street_image, kp_image, dead_end_sign, kp_dead_end, matches_dead_end[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 7. 显示结果
    cv2.imshow("Attention Sign Matches", result_attention)
    cv2.imshow("Dead End Sign Matches", result_dead_end)

    # 8. 显示标志位置
    if len(matches_attention) > 5:  # 如果有足够的匹配
        attention_end_time = time.time()
        attention_time = (attention_end_time - attention_start_time) * 1000  # 毫秒
        print(f"Attention sign found in: {attention_time:.2f} ms")

    if len(matches_dead_end) > 5:  # 如果有足够的匹配
        dead_end_end_time = time.time()
        dead_end_time = (dead_end_end_time - dead_end_start_time) * 1000  # 毫秒
        print(f"Dead end sign found in: {dead_end_time:.2f} ms")

    # 9. 如果没有找到标志
    if len(matches_attention) <= 5 and len(matches_dead_end) <= 5:
        cv2.putText(street_image, "No sign found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # 10. 显示最终结果
    cv2.imshow("Detected Road Signs", street_image)
    cv2.waitKey(0)  # 按键继续到下一图像

cv2.destroyAllWindows()  # 关闭所有窗口
