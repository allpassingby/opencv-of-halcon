import numpy as np
from sklearn.preprocessing import MinMaxScaler


"""
最小-最大归一化Min-Max Normalization :RGB图像的像素值从[0, 255]转换到[0, 1]。 
 特征值的范围已知且没有离群点时，适合使用该方法
优点：数据范围被压缩到[0, 1]，使得优化算法（如梯度下降）更容易收敛
缺点：对离群点敏感，离群点可能导致归一化后的数据不在[0, 1]区间内。
"""
# 假设数据
data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

# 使用MinMaxScaler进行最小-最大归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

print("最小-最大归一化后的数据:")
print(data_scaled)

"""
Z-score标准化Standardization : 图像每个像素的值会根据该像素所在的通道（如R、G、B通道）的均值和标准差进行变换
当数据的分布不明确时，使用Z-score标准化更加稳健
优点：使得所有特征都按相同的尺度进行训练，有助于提高算法的性能，尤其在基于距离的算法中
缺点：结果并不限定在固定的范围内（如[0, 1]），在一些对数据范围有要求的任务中可能不适用
"""
from sklearn.preprocessing import StandardScaler
# 假设数据
data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

# 使用StandardScaler进行标准化
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

print("Z-score标准化后的数据:")
print(data_standardized)

"""
 均值归一化（Mean Normalization）: 数据的范围在[-1, 1]之间，适合对称分布的应用
优点：数据的范围在[-1, 1]之间，适合对称分布的应用
缺点：对于离群点较为敏感
"""

import numpy as np

# 假设数据
data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

# 计算均值和最大值、最小值
mean = np.mean(data, axis=0)
min_value = np.min(data, axis=0)
max_value = np.max(data, axis=0)

# 均值归一化
data_mean_normalized = (data - mean) / (max_value - min_value)

print("均值归一化后的数据:")
print(data_mean_normalized)


"""
L2 范数归一化（L2 Normalization）:常用于图像特征提取，特别是在计算余弦相似度时
 L2范数归一化不会改变数据的方向，只会缩放向量的大小。
L2归一化通常用于：在机器学习中，特别是在使用支持向量机（SVM）、神经网络和k-近邻算法（KNN）时，确保所有的特征具有相同的权重。
在计算图像、文本或特征向量的相似度时（如余弦相似度），通过归一化可以使得每个向量的大小一致，避免某些特征因尺度过大而主导相似度计算
优点：数据的范围在[-1, 1]之间，适合对称分布的应用
缺点：对于具有显著不同尺度的特征，L2归一化可能会使得原本重要的特征的影响被忽视
"""

arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])

# 计算每行的L2范数
l2_norm = np.linalg.norm(arr, axis=1, keepdims=True)

# L2归一化
normalized_arr = arr / l2_norm

print("L2归一化结果：")
print(normalized_arr)

import cv2
import numpy as np
import matplotlib.pyplot as plt




"""
局部归一化：使用局部归一化可以增强图像中某些区域的特征，特别是在图像细节比较重要时
 L2范数归一化不会改变数据的方向，只会缩放向量的大小。
L2归一化通常用于：在机器学习中，特别是在使用支持向量机（SVM）、神经网络和k-近邻算法（KNN）时，确保所有的特征具有相同的权重。
在计算图像、文本或特征向量的相似度时（如余弦相似度），通过归一化可以使得每个向量的大小一致，避免某些特征因尺度过大而主导相似度计算
优点：数据的范围在[-1, 1]之间，适合对称分布的应用
缺点：对于具有显著不同尺度的特征，L2归一化可能会使得原本重要的特征的影响被忽视
"""
def local_normalization(image, window_size=5):
    """在图像上进行局部归一化，使用每个像素邻域的均值和标准差进行归一化"""
    height, width = image.shape[:2]
    result = np.zeros_like(image, dtype=np.float32)

    # 遍历每个像素
    for y in range(window_size // 2, height - window_size // 2):
        for x in range(window_size // 2, width - window_size // 2):
            # 取窗口区域
            patch = image[y - window_size // 2: y + window_size // 2 + 1,
                    x - window_size // 2: x + window_size // 2 + 1]
            # 计算窗口的均值和标准差
            mean = np.mean(patch)
            std = np.std(patch)

            # 归一化
            result[y, x] = (image[y, x] - mean) / (std + 1e-5)  # 避免除以0

    return result


# 加载图片
image = cv2.imread('left.png', cv2.IMREAD_GRAYSCALE)

# 应用局部归一化
normalized_image = local_normalization(image)

# 显示归一化后的图像
plt.imshow(normalized_image, cmap='gray')
plt.title('Local Normalization')
plt.axis('off')
plt.show()


"""
Batch Normalization（BN）则是在神经网络训练中进行的标准化，主要用来加速训练、提高稳定性和减少过拟合，广泛应用于深度学习的各种任务中，如图像分类、目标检测、生成对抗网络等。
通过可学习的参数 
𝛾
γ 和 
𝛽
β 来调整归一化后的数据 对应尺度和偏置
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 创建一个简单的卷积神经网络（CNN）模型，包含 Batch Normalization
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 一个卷积层，输出通道数为16，卷积核大小为3x3
        self.conv1 = nn.Conv2d(1, 16, 3)
        # Batch Normalization 层
        self.bn1 = nn.BatchNorm2d(16)

        # 一个全连接层
        self.fc1 = nn.Linear(16 * 26 * 26, 10)

    def forward(self, x):
        x = self.conv1(x)
        # Batch Normalization
        x = self.bn1(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.fc1(x)
        return x


# 假设我们有一个批次的灰度图像数据（batch_size=8, channel=1, height=28, width=28）
batch_data = torch.randn(8, 1, 28, 28)

# 创建模型
model = SimpleCNN()

# 前向传播
output = model(batch_data)

print("Output shape after BN:", output.shape)
