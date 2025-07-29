import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# 最小二乘法拟合圆函数
def fit_circle_ls(xs, ys):
    """最小二乘圆拟合，返回 (xc, yc, r)."""
    def fun(c):
        x0, y0, r = c
        return np.sqrt((xs-x0)**2 + (ys-y0)**2) - r
    # 初始猜测：圆心平均、半径平均
    x_m, y_m = xs.mean(), ys.mean()
    r0 = np.mean(np.sqrt((xs-x_m)**2 + (ys-y_m)**2))
    c0 = [x_m, y_m, r0]
    c_opt, _ = optimize.leastsq(fun, c0)
    return c_opt  # xc, yc, r

# 1. 读图、灰度、平滑
img = cv2.imread('E:/ruanjian/HALCON-25.05-Progress/cs/jiancequyu/09-15-42-4537-00.jpg')
gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5,5), 0)

# 2. 卡尺参数
center_est_1 = (1224, 985)      # 第一个圆的初步估计圆心位置
radius_est_1 = 799              # 第一个圆的初步估计半径
center_est_2 = (1224, 985)      # 第二个圆的初步估计圆心位置
radius_est_2 = 680              # 第二个圆的初步估计半径
num_calipers = 60               # 布置 60 把卡尺
caliper_len = 40                # 每把卡尺长度（像素）
caliper_half = caliper_len // 2

# 存储所有圆的边缘点
edge_pts_1 = []
edge_pts_2 = []

# 3. 沿圆周布置卡尺
angles = np.linspace(0, 2*np.pi, num_calipers, endpoint=False)

# 处理第一个圆
for a in angles:
    # 卡尺中心点：在估计圆周上
    cx = center_est_1[0] + radius_est_1 * np.cos(a)
    cy = center_est_1[1] + radius_est_1 * np.sin(a)
    # 卡尺方向（法线方向），单位向量
    nx, ny = np.cos(a), np.sin(a)
    # 卡尺线上的采样点
    samples = []
    for t in np.linspace(-caliper_half, caliper_half, caliper_len):
        x = cx + t * nx
        y = cy + t * ny
        # 双线性插值获取灰度
        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
            samples.append(gray[int(y), int(x)])
        else:
            samples.append(0)
    # 梯度幅值
    grad = np.abs(np.convolve(samples, [1,0,-1], mode='same'))
    # 找最大梯度位置
    idx = np.argmax(grad)
    # 对应的边缘坐标
    ex = cx + (idx - caliper_half) * nx
    ey = cy + (idx - caliper_half) * ny
    edge_pts_1.append((ex, ey))

# 处理第二个圆
for a in angles:
    # 卡尺中心点：在估计圆周上
    cx = center_est_2[0] + radius_est_2 * np.cos(a)
    cy = center_est_2[1] + radius_est_2 * np.sin(a)
    # 卡尺方向（法线方向），单位向量
    nx, ny = np.cos(a), np.sin(a)
    # 卡尺线上的采样点
    samples = []
    for t in np.linspace(-caliper_half, caliper_half, caliper_len):
        x = cx + t * nx
        y = cy + t * ny
        # 双线性插值获取灰度
        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
            samples.append(gray[int(y), int(x)])
        else:
            samples.append(0)
    # 梯度幅值
    grad = np.abs(np.convolve(samples, [1,0,-1], mode='same'))
    # 找最大梯度位置
    idx = np.argmax(grad)
    # 对应的边缘坐标
    ex = cx + (idx - caliper_half) * nx
    ey = cy + (idx - caliper_half) * ny
    edge_pts_2.append((ex, ey))

# 最小二乘拟合第一个圆
edge_pts_1 = np.array(edge_pts_1)
xs1, ys1 = edge_pts_1[:,0], edge_pts_1[:,1]
xc1, yc1, r1 = fit_circle_ls(xs1, ys1)

# 最小二乘拟合第二个圆
edge_pts_2 = np.array(edge_pts_2)
xs2, ys2 = edge_pts_2[:,0], edge_pts_2[:,1]
xc2, yc2, r2 = fit_circle_ls(xs2, ys2)

# 5. 可视化
out = img.copy()
# 第一个圆——灰色
cv2.circle(out, (int(center_est_1[0]), int(center_est_1[1])), int(radius_est_1), (200, 200, 200), 1)
# 拟合第一个圆——绿色
cv2.circle(out, (int(xc1), int(yc1)), int(r1), (0, 255, 0), 2)
# 第二个圆——灰色
cv2.circle(out, (int(center_est_2[0]), int(center_est_2[1])), int(radius_est_2), (200, 200, 200), 1)
# 拟合第二个圆——蓝色
cv2.circle(out, (int(xc2), int(yc2)), int(r2), (0, 0, 255), 2)
# 绘制第一个圆的测量点——红点
for (ex, ey) in edge_pts_1:
    cv2.circle(out, (int(ex), int(ey)), 3, (0, 0, 255), -1)
# 绘制第二个圆的测量点——红点
for (ex, ey) in edge_pts_2:
    cv2.circle(out, (int(ex), int(ey)), 3, (0, 0, 255), -1)

# 显示图像
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
plt.title("模拟卡尺找圆 (Caliper Circle Fit) - 双圆")
plt.axis('off')
plt.tight_layout()
plt.show()
