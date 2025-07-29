
from matplotlib import rcParams
import matplotlib.font_manager as fm

my_font = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')  # 微软雅黑路径
rcParams['font.family'] = my_font.get_name()

"""
# 读取图像并转换为灰度图
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("02.jpg")  # 换成你的图像路径
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ========== 1. Sobel 边缘检测 ==========
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # X方向
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Y方向
sobel_combined = cv2.magnitude(sobel_x, sobel_y)

# ========== 2. Laplacian 边缘检测 ==========
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# ========== 3. Canny 边缘检测 ==========
canny = cv2.Canny(gray, 100, 200)

# ========== 显示结果 ==========
titles = ['Original Gray', 'Sobel X', 'Sobel Y', 'Sobel Combined', 'Laplacian', 'Canny']
images = [gray, sobel_x, sobel_y, sobel_combined, laplacian, canny]

plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

"""

"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm

my_font = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')  # 微软雅黑路径
rcParams['font.family'] = my_font.get_name()

# 解决负号显示问题
rcParams['axes.unicode_minus'] = False

# 读取图像
img = cv2.imread('street.jpg')  # 👈 替换为你的图片路径
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 添加一些噪声（模拟场景）
noise = np.random.randint(0, 50, img.shape, dtype='uint8')
noisy_img = cv2.add(img, noise)

# 1. 均值滤波（平均值滤波）
blur_mean = cv2.blur(noisy_img, (5, 5))

# 2. 高斯滤波（加权平均）
blur_gaussian = cv2.GaussianBlur(noisy_img, (5, 5), 1)

# 3. 中值滤波（对椒盐噪声更有效）
blur_median = cv2.medianBlur(noisy_img, 5)

# 4. 双边滤波（保边去噪）
blur_bilateral = cv2.bilateralFilter(noisy_img, d=9, sigmaColor=75, sigmaSpace=75)

# 展示所有结果
titles = ['原图 + 噪声', '均值滤波', '高斯滤波', '中值滤波', '双边滤波']
images = [noisy_img, blur_mean, blur_gaussian, blur_median, blur_bilateral]

plt.figure(figsize=(15, 6))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
"""


""" 角点检测算子 
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("street.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Harris角点
harris = cv2.cornerHarris(np.float32(gray), blockSize=2, ksize=3, k=0.04)
harris_img = img.copy()
harris_img[harris > 0.01 * harris.max()] = [0, 0, 255]  # 红点标角

# Shi-Tomasi角点（又叫 GFTT）
shi_corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
shi_img = img.copy()
for pt in shi_corners:
    x, y = pt.ravel()
    cv2.circle(shi_img, (int(x), int(y)), 3, (0, 255, 0), -1)

# FAST角点
fast = cv2.FastFeatureDetector_create()
fast_kps = fast.detect(gray, None)
fast_img = cv2.drawKeypoints(img, fast_kps, None, color=(255, 0, 0))

# ORB角点
orb = cv2.ORB_create()
orb_kps = orb.detect(gray, None)
orb_img = cv2.drawKeypoints(img, orb_kps, None, color=(255, 255, 0))

# 显示结果
titles = ["Harris角点", "Shi-Tomasi角点", "FAST角点", "ORB角点"]
images = [harris_img, shi_img, fast_img, orb_img]

plt.figure(figsize=(14, 8))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
"""

""" 角点检测
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像，转为灰度并二值化
img = cv2.imread("02.jpg", 0)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 定义结构元素（核），可改为椭圆或十字
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 各类操作
dilate = cv2.dilate(binary, kernel, iterations=1)
erode = cv2.erode(binary, kernel, iterations=1)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)

# 展示
titles = ["原图", "膨胀", "腐蚀", "开运算", "闭运算", "梯度", "顶帽", "黑帽"]
images = [binary, dilate, erode, opening, closing, gradient, tophat, blackhat]

plt.figure(figsize=(12, 8))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
"""

""" 特征检测算法
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# 读取图像
img = cv2.imread('street.jpg')
if img is None:
    raise FileNotFoundError("无法读取 'example.jpg'，请检查路径或图像是否存在。")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. SIFT 特征
sift = cv2.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(gray, None)
img_sift = cv2.drawKeypoints(img, kp_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 2. ORB 特征
orb = cv2.ORB_create()
kp_orb, des_orb = orb.detectAndCompute(gray, None)
img_orb = cv2.drawKeypoints(img, kp_orb, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# 4. HOG 特征（可视化梯度图）
win_size = (64, 128)
hog = cv2.HOGDescriptor()
hog_image = cv2.resize(gray, win_size)
hog_feats = hog.compute(hog_image)
hog_feats_vis = hog_image.copy()
cv2.putText(hog_feats_vis, "HOG特征提取成功", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255), 1)

# 5. LBP 特征（局部二值模式）
radius = 1
n_points = 8 * radius
lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

# 展示所有特征图
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title("原图")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("SIFT")
plt.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("ORB")
plt.imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
plt.axis('off')


plt.subplot(2, 3, 5)
plt.title("HOG 输入图")
plt.imshow(hog_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("LBP 图")
plt.imshow(lbp, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
"""

""" SVM 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# 1. 加载数据（这里使用内置鸢尾花数据）
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 2. 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 定义并训练 SVM 模型
clf = SVC(kernel='linear')  # 可改为 'rbf'、'poly'、'sigmoid'
clf.fit(X_train, y_train)

# 5. 预测与评估
y_pred = clf.predict(X_test)
print("准确率：", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
"""

"""LSTm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 构造正弦波序列数据
def generate_sin_wave(seq_length, num_samples):
    X, y = [], []
    for _ in range(num_samples):
        start = np.random.rand() * 2 * np.pi
        xs = np.linspace(start, start + 2 * np.pi, seq_length + 1)
        data = np.sin(xs)
        X.append(data[:-1])
        y.append(data[1:])
    return np.array(X), np.array(y)

# LSTM 网络结构定义
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out)

# 超参数
seq_length = 50
num_samples = 200
input_size = 1
hidden_size = 32
num_layers = 1
num_epochs = 100
lr = 0.01

# 数据准备
X, y = generate_sin_wave(seq_length, num_samples)
X = torch.tensor(X[:, :, None], dtype=torch.float32)
y = torch.tensor(y[:, :, None], dtype=torch.float32)

# 初始化模型和训练组件
model = SimpleLSTM(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练过程
for epoch in range(num_epochs):
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 预测结果
predicted = model(X).detach().numpy()

# 画出其中一个序列的真实值与预测值
plt.figure(figsize=(10, 4))
plt.plot(y[0].numpy(), label='True')
plt.plot(predicted[0], label='Predicted')
plt.title("LSTM Sine Wave Prediction")
plt.xlabel("Time Step")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
"""


"""  L1 l2 正则化 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge
from sklearn.datasets import make_regression

# 生成模拟回归数据
X, y, coef = make_regression(n_samples=100, n_features=20, n_informative=5,
                             noise=10, coef=True, random_state=42)

# L1正则化（Lasso）
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

# L2正则化（Ridge）
ridge = Ridge(alpha=0.1)
ridge.fit(X, y)

# 绘制对比图
plt.figure(figsize=(12, 6))
plt.plot(coef, 'ko', label='True Coefficients')
plt.plot(lasso.coef_, 'ro', label='L1 (Lasso)')
plt.plot(ridge.coef_, 'bo', label='L2 (Ridge)')
plt.axhline(0, color='gray', linestyle='--')
plt.title('L1 vs L2 Regularization on Feature Coefficients')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""