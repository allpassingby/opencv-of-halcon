import easyocr
import cv2
import matplotlib.pyplot as plt

# 初始化 EasyOCR 识别器，指定要识别的语言
reader = easyocr.Reader(['en'])  # 'en' 表示英语，如果要识别其他语言，可以在这里添加

# 读取图像文件
image_path = 'chars_training_05.png'  # 请将这里的 image.png 替换为你自己的图像路径
img = cv2.imread(image_path)

# 使用 EasyOCR 识别图像中的文字
result = reader.readtext(image_path)

# 绘制识别结果
for detection in result:
    # detection[0] 包含识别区域的坐标
    # detection[1] 包含识别的文本内容
    # detection[2] 包含文本的可信度得分
    top_left = tuple(detection[0][0])  # 左上角坐标
    bottom_right = tuple(detection[0][2])  # 右下角坐标
    text = detection[1]  # 识别出的文本内容

    # 在图像上绘制框和文本
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)  # 绘制矩形框
    cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # 绘制文本

# 显示处理后的图像
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # OpenCV 默认以 BGR 格式加载图像，需要转换为 RGB 才能正确显示
plt.axis('off')  # 关闭坐标轴
plt.show()

# 输出识别结果
for detection in result:
    print(f"Detected text: {detection[1]} with confidence: {detection[2]}")
