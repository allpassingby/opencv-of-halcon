# **Opencv之Halcon开发复现**



在现代计算机视觉应用中，OpenCV 和 Halcon 都是广泛使用的图像处理和机器视觉库。**OpenCV**（开源计算机视觉库）为图像处理提供了大量工具，适用于实时计算机视觉应用。它通过高度优化的函数和强大的社区支持，广泛应用于从基础图像处理到复杂的机器学习任务。**Halcon** 是一款商业化的机器视觉软件，主要用于工业自动化领域，提供强大的图像处理功能，特别适合在高精度和高可靠性要求的场景中应用。



#### 本文记录了在实习过程中的学习与探索，旨在分享我在使用 **OpenCV** 和 **Halcon** 进行图像处理和机器视觉开发中的一些经验与方法。提供的技术方案和方法不一定是最优解，在不同的应用场景中可能需要根据具体需求进行调整和优化。KZ-W



## NO.01  卡尺找圆

对于同心圆结构的工业零件（如：轴承、垫圈、法兰盘）提取有效区域进行缺陷检测 。通过模拟卡尺方法，沿着圆的周围布置卡尺并检测圆的边缘。然后使用 **最小二乘法拟合圆**，得到圆的准确位置和半径。



Halcon代码：

```halcon
* Step 1: 读取图像并转灰度
read_image(Image, 'E:/ruanjian/HALCON-25.05-Progress/cs/jiancequyu/09-15-42-4537-00.jpg')
rgb1_to_gray(Image, GrayImage)

* Step 2: 创建计量模型
create_metrology_model(MetrologyHandle)

* Step 3: 添加第一个大圆的测量模型（中心坐标和半径你可微调）
* 参数顺序：Row, Column, Radius, Length1, Length2, Threshold, NumMeasures
add_metrology_object_circle_measure(MetrologyHandle, 985.7, 1224.0, 799.0, 80, 5, 1, 40, [], [], Index1)

* Step 4: 添加第二个较小外圆的测量模型
add_metrology_object_circle_measure(MetrologyHandle, 985.0, 1224.0, 680.0, 80, 5, 1, 30, [], [], Index2)

* Step 5: 应用测量模型到图像
apply_metrology_model(GrayImage, MetrologyHandle)

* Step 6: 获取测量结果轮廓（显示用）
get_metrology_object_result_contour(Contour1, MetrologyHandle, Index1, 'all', 1.5)
get_metrology_object_result_contour(Contour2, MetrologyHandle, Index2, 'all', 1.5)

* Step 7: 显示图像与两个圆形结果
get_image_size(Image, Width, Height)
dev_open_window(0, 0, Width/2, Height/2, 'black', WindowID)
dev_display(Image)
dev_set_color('green')
dev_display(Contour1)
dev_set_color('blue')
dev_display(Contour2)

* Step 8: 获取参数（可选）
get_metrology_object_result(MetrologyHandle, Index1, 'all', 'result_type', 'all_param', Param1)
get_metrology_object_result(MetrologyHandle, Index2, 'all', 'result_type', 'all_param', Param2)

* Step 9: 释放资源
clear_metrology_model(MetrologyHandle)

```



opencv复现:基于卡尺边缘提取与最小二乘拟合 对工业图像中同心圆结构的高精度识别与检测区域提取。

```python
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

```



![kczy](F:\常州易管\kczy.png)





# **NO.02 判断物体方向**



通过图像处理算法提取标签区域，计算标签的旋转角度，判断判断标签指向。通过图像处理技术，从输入图像中提取并检测箭头，首先通过阈值化、开运算和轮廓查找方法确定箭头的位置和形状，然后通过计算最小外接矩形和质心，判断箭头的方向。

## halcon代码

```halcon
* 读取图像
read_image(Image, 'E:/ruanjian/HALCON-25.05-Progress/cs/biaoqian/left.png')

* gen_region_runs (ROI_0, [230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,287,288,288,289,289,290,291,292,293,294,295,296,297], [448,448,448,448,448,448,447,447,447,447,447,447,447,447,447,447,447,447,446,446,446,446,446,446,446,446,446,446,446,446,445,445,445,445,445,445,445,445,445,445,445,445,444,444,444,444,444,444,444,444,444,444,444,444,443,443,443,384,443,384,443,384,415,384,384,384,384,384,384,384,384], [476,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,506,384,506,385,506,385,506,506,506,506,506,506,485,445,404])

* erosion_circle (ROI_0, RegionErosion, 5)

* 打开一个窗口并初始化WindowID
dev_open_window(0, 0, 600, 400, 'black', WindowID)

* 显示图像以确认加载正确
disp_image(Image, WindowID)

* 转换为灰度图像
rgb1_to_gray(Image, GrayImage)

* 显示灰度图像以确认转换成功
disp_image(GrayImage, WindowID)

* 使用阈值进行二值化处理
threshold(GrayImage, Region, 180, 255)
* 
opening_circle (Region, RegionOpening, 5)
connection (RegionOpening, ConnectedRegions)


* count_obj (ConnectedRegions, Number)
* for Index := 1 to Number by 1
*     select_obj (ConnectedRegions, ObjectSelected,Index)
* endfor


dev_clear_window ()
* dev_display (Image)
select_shape (ConnectedRegions, SelectedRegions, 'area', 'and', 7000, 17000)

* area_center (SelectedRegions, Area, Row, Column)
* move_region (SelectedRegions, RegionMoved, 0, 5)
* difference (SelectedRegions, RegionMoved, RegionDifference)
* opening_circle (RegionDifference, RegionOpening1, 1)


* difference (RegionMoved, SelectedRegions, RegionDifference)

* dilation_circle (SelectedRegions, RegionDilation, 15)
reduce_domain (GrayImage,  SelectedRegions, ImageReduced)

dev_display (SelectedRegions)
* 查找箭头的轮廓

edges_sub_pix(ImageReduced, Edges, 'canny', 3, 20, 40)
* closing_rectangle1(Gra, RegionClosed, 5, 5)


* 计算轮廓的方向
orientation_region(ImageReduced, Orientation)

if (Orientation < 0)
    * 箭头指向右边（向右）
    dev_disp_text('向右', 'window', 12, 12, 'black', [], [])
else
    * 箭头指向左边（向左）
    dev_disp_text('向左', 'window', 12, 12, 'black', [], [])
endif


```

## opencv复现

```
import cv2
import numpy as np

# 读取图像
image = cv2.imread('E:/ruanjian/HALCON-25.05-Progress/cs/biaoqian/left.png')

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

```





![fx](F:\常州易管\fx.png)

## NO.03 给定区域二值化

threshold(GrayImage, Region, 180, 255)  

简单来说，Halcon的threshold函数是获取区间[a, b]之间的灰度值，OpenCV的threshold只能针对大于或者小于a或者b的灰度值处理，一个是双阈值，一个是单阈值。

```python
import cv2
# 方法一：cv2.inRange(gray, 180, 255)
# 用opencv 复现 threshold(GrayImage, Region, 180, 255)

img = cv2.imread("123.png", cv2.IMREAD_GRAYSCALE)
gray = img  # 直接使用灰度图

cv2.imshow('src', img)

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 第一次阈值分割：低阈值
ret1, thres_low = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
cv2.imshow('thres_low', thres_low)

# 第二次阈值分割：高阈值
ret2, thres_high = cv2.threshold(gray, 255, 255, cv2.THRESH_BINARY)
cv2.imshow('thres_high', thres_high)

# 两次阈值图像相减，保留中间灰度段的像素区域
threshold = thres_low - thres_high
cv2.imshow('threshold', threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()
```



工作日历：视觉成像 海康MVS 分割分类 maskrcnn 

 



# NO.04 判断零部件的正反

生成圆形掩码提取兴趣区域。接着，裁剪出零部件的矩形区域，并转换为灰度图像，计算该区域的平均亮度。根据亮度值与预设阈值对比，判断零部件的正反面。具体如下：通过以下步骤判断图像是正面还是背面：首先通过圆形 ROI 提取图像的一部分，并对这部分图像进行灰度化处理；然后计算图像的平均亮度，如果亮度较高，判断为正面，反之则为背面。最后，代码在图像上显示判断结果。该方法适用于通过亮度差异来判断图像正反面，常用于带有标识或有明显亮度变化的图像处理中。

## halcon代码

```halcon
* read_image(Image, 'E:/ruanjian/HALCON-25.05-Progress/cs/zhengfan/14-08-59-5358-00.jpg')
read_image(Image, 'E:/ruanjian/HALCON-25.05-Progress/cs/zhengfan/15-19-00-9330-00.jpg')
* Image Acquisition 01: Code generated by Image Acquisition 01
list_files ('E:/ruanjian/HALCON-25.05-Progress/cs/zhengfan', ['files','follow_links'], ImageFiles)
tuple_regexp_select (ImageFiles, ['\\.(tif|tiff|gif|bmp|jpg|jpeg|jp2|png|pcx|pgm|ppm|pbm|xwd|ima|hobj)$','ignore_case'], ImageFiles)
for Index := 0 to |ImageFiles| - 1 by 1
    read_image (Image, ImageFiles[Index])
    * Image Acquisition 01: Do something
    
    
    
    gen_circle (ROI_0, 991.356, 1067.11, 265.363)
    
    
get_image_size(Image,width,height)
* dev_open_window(0, 0, 600, 400, 'black', WindowID)
* disp_image(Image, WindowID)


Row1 := 645
Column1 := 710
Row2 := 1433
Column2 := 1500
gen_rectangle1(RectangleRegion, Row1, Column1, Row2, Column2)


reduce_domain(Image, RectangleRegion, Image)
* clear_window(WindowID)


* disp_image(ImageRegion, WindowID)

rgb1_to_gray(Image, GrayImage)

intensity (GrayImage, GrayImage, Mean, MeanBrightness)

* write_string(WindowID, MeanBrightness)

if (MeanBrightness > 58) 
    Direction := '正面'
else
    Direction := '背面'
endif


* 输出判断结果
* write_string(WindowID,Direction)
dev_display (Image)
* close_window(WindowID)
    stop ()
endfor

```

## opencv复现：

```python
import cv2
import numpy as np

# 指定单张图像路径
image_path = 'E:/ruanjian/HALCON-25.05-Progress/cs/zhengfan/15-19-00-9330-00.jpg'

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

# 将图像调整为适合显示的大小，避免显示过大
resize_factor = 0.5  # 控制缩放比例
resized_image = cv2.resize(image, (int(width * resize_factor), int(height * resize_factor)))

# 在图像上添加方向文本
cv2.putText(resized_image, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# 设置显示窗口大小
cv2.namedWindow(f'Processed Image - {image_path}', cv2.WINDOW_NORMAL)
cv2.resizeWindow(f'Processed Image - {image_path}', 800, 600)  # 设置窗口大小为 800x600

# 显示图像和结果
cv2.imshow(f'Processed Image - {image_path}', resized_image)

# 等待用户按键后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

```



![zf](F:\常州易管\zf.png)



## **NO.05 网格缺陷检测**

 Halcon中对应的例子为novelty_detection_dyn_threshold.hdev 用于检测网格缺陷 实现步骤：动态二值化，区域面积筛选。具体：读取并灰度化图像，再通过高斯模糊平滑图像。接着，使用两次阈值操作进行二值化，提取出可能的缺陷区域。然后，查找图像中的轮廓，并根据轮廓的面积判断是否为缺陷区域。

halcon代码：

```

dev_update_window ('off')
read_image (Image, 'plastic_mesh/plastic_mesh_01')
dev_close_window ()
get_image_size (Image, Width, Height)
dev_open_window_fit_image (Image, 0, 0, Width, Height, WindowHandle)
set_display_font (WindowHandle, 18, 'mono', 'true', 'false')
dev_set_draw ('margin')
dev_set_line_width (3)
* Each of the images is read and smoothed. Subsequently
* dyn_threshold is performed and connected regions are
* looked for. The parameter 'area' of the operator select_shape
* makes it possible to find regions that differ in the area
* size. Found errors are finally counted and displayed.
for J := 1 to 14 by 1
    read_image (Image, 'plastic_mesh/plastic_mesh_' + J$'02')
    mean_image (Image, ImageMean, 49, 49)
    dyn_threshold (Image, ImageMean, RegionDynThresh, 5, 'dark')
    connection (RegionDynThresh, ConnectedRegions)
    select_shape (ConnectedRegions, ErrorRegions, 'area', 'and', 500, 99999)
    count_obj (ErrorRegions, NumErrors)
    dev_display (Image)
    dev_set_color ('red')
    dev_display (ErrorRegions)
    * If the number of errors exceeds zero, the message 'Mesh not
    * OK' is displayed. Otherwise the web is undamaged
    * and 'Mesh OK' is displayed.
    if (NumErrors > 0)
        disp_message (WindowHandle, 'Mesh not OK', 'window', 24, 12, 'black', 'true')
    else
        disp_message (WindowHandle, 'Mesh OK', 'window', 24, 12, 'black', 'true')
    endif
    * If the sequence number of the image to be inspected is
    * lower than 14, the request to press 'Run' to continue appears.
    * If the last image is read, pressing 'Run' will clear the SVM.
    if (J < 14)
        disp_continue_message (WindowHandle, 'black', 'true')
        stop ()
    endif
endfor
```





opencv复现：

```python
import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX

img = cv2.imread('plastic_mesh_13.png')
cv2.imshow('src',img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3,3), 1.0)

#ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
_,thresh1 = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY_INV )
_,thresh2 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV )
thresh = thresh1 - thresh2
#cv2.imshow('thresh',thresh)

contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

isNG = False
for cnt in contours:
  area = cv2.contourArea(cnt)
  if(area>350):
    cv2.drawContours(img,cnt,-1,(0,0,255),2)
    isNG = True
#  else:
#    cv2.drawContours(img,cnt,-1,(0,255,0),1)

if isNG:
  rect, basline = cv2.getTextSize('Mesh Not OK', font, 1.0, 2)
  cv2.rectangle(img, (10,10,int(rect[0]*0.7),rect[1]), (212, 233, 252), -1, 8)
  cv2.putText(img,'Mesh Not OK', (10,5+rect[1]), font, 0.7, (0,0,255), 2)
else:
  rect, basline = cv2.getTextSize('Mesh OK', font, 1.0, 2)
  cv2.rectangle(img, (10,10,int(rect[0]*0.7),rect[1]), (212, 233, 252), -1, 8)
  cv2.putText(img,'Mesh OK', (10,5+rect[1]), font, 0.7, (0,200,0), 2)
cv2.imshow('meshDefects', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```





![wgqx](F:\常州易管\wgqx.png)



# NO.06 图像拼接

首先读取两张待拼接的图像并将其转换为灰度图像。接着，使用 ORB 算法提取图像中的关键点和描述符，并通过暴力匹配（BFMatcher）方法匹配两张图像的特征点。然后，利用 RANSAC 算法计算单应矩阵（Homography），从而估计两张图像之间的透视变换关系。接下来，使用计算得到的单应矩阵对图像进行透视变换，并将变换后的图像与原图拼接在一起，生成最终的全景图。

halcon代码

```halcon

* Read and display the images
read_image (Image1, 'mosaic/building_01')
read_image (Image2, 'mosaic/building_02')
get_image_size (Image1, Width, Height)
dev_close_window ()
dev_open_window (0, 0, Width, Height, 'black', WindowHandle)
set_display_font (WindowHandle, 16, 'mono', 'true', 'false')
dev_display (Image1)
disp_message (WindowHandle, 'Image 1 to be matched', 'window', 12, 12, 'black', 'true')
disp_continue_message (WindowHandle, 'black', 'true')
stop ()
dev_display (Image2)
disp_message (WindowHandle, 'Image 2 to be matched', 'window', 12, 12, 'black', 'true')
disp_continue_message (WindowHandle, 'black', 'true')
stop ()
* 
* Set the zoom factor by which the images are zoomed down to obtain
* initial estimates of the projective transformation and of the radial
* distortion coefficient.
Factor := 0.5
* Zoom down the images.
zoom_image_factor (Image1, Image1Zoomed, Factor, Factor, 'constant')
zoom_image_factor (Image2, Image2Zoomed, Factor, Factor, 'constant')
* Extract points to be matched from the low-resolution images.
points_foerstner (Image1Zoomed, 1, 2, 3, 50, 0.1, 'gauss', 'true', Rows1, Cols1, CoRRJunctions, CoRCJunctions, CoCCJunctions, RowArea, ColumnArea, CoRRArea, CoRCArea, CoCCArea)
points_foerstner (Image2Zoomed, 1, 2, 3, 50, 0.1, 'gauss', 'true', Rows2, Cols2, CoRRJunctions, CoRCJunctions, CoCCJunctions, RowArea, ColumnArea, CoRRArea, CoRCArea, CoCCArea)
* Read out the image size, which is only used here to make the search
* range for point matches as large as possible.
get_image_size (Image1Zoomed, Width, Height)
* Perform the initial point matching on the low-resolution images.
proj_match_points_distortion_ransac (Image1Zoomed, Image2Zoomed, Rows1, Cols1, Rows2, Cols2, 'ncc', 10, 0, 0, Height, Width, 0, 0.5, 'gold_standard', 1, 42, HomMat2D, Kappa, Error, Points1, Points2)
* Adapt the projective transformation to the original image resolution.
hom_mat2d_scale_local (HomMat2D, Factor, Factor, HomMat2DGuide)
hom_mat2d_scale (HomMat2DGuide, 1.0 / Factor, 1.0 / Factor, 0, 0, HomMat2DGuide)
* Adapt the radial distortion coefficient to the original image resolution.
KappaGuide := Kappa * Factor * Factor
* Extract points to be matched from the original images.
points_foerstner (Image1, 1, 2, 3, 50, 0.1, 'gauss', 'true', Rows1, Cols1, CoRRJunctions, CoRCJunctions, CoCCJunctions, RowArea, ColumnArea, CoRRArea, CoRCArea, CoCCArea)
points_foerstner (Image2, 1, 2, 3, 50, 0.1, 'gauss', 'true', Rows2, Cols2, CoRRJunctions, CoRCJunctions, CoCCJunctions, RowArea, ColumnArea, CoRRArea, CoRCArea, CoCCArea)
* Match the points in the original images using the estimates of the
* projective transformation and of the radial distortion coefficient as
* a guide.
proj_match_points_distortion_ransac_guided (Image1, Image2, Rows1, Cols1, Rows2, Cols2, 'ncc', 10, HomMat2DGuide, KappaGuide, 5, 0.5, 'gold_standard', 1, 42, HomMat2D, Kappa, Error, Points1, Points2)
* Construct camera parameters for the purpose of rectifying the images,
* i.e., to remove the radial distortions.
get_image_size (Image1, Width, Height)
gen_cam_par_area_scan_telecentric_division (1.0, Kappa, 1.0, 1.0, 0.5 * (Width - 1), 0.5 * (Height - 1), Width, Height, CamParDist)
* Remove the radial distortions from the images.
change_radial_distortion_cam_par ('fixed', CamParDist, 0, CamPar)
change_radial_distortion_image (Image1, Image1, Image1Rect, CamParDist, CamPar)
change_radial_distortion_image (Image2, Image2, Image2Rect, CamParDist, CamPar)
* Construct a mosaic from the two rectified images.  Note that the images
* fit together perfectly.
concat_obj (Image1Rect, Image2Rect, ImagesRect)
gen_projective_mosaic (ImagesRect, MosaicImage, 1, 1, 2, HomMat2D, 'default', 'false', MosaicMatrices2D)
get_image_size (MosaicImage, Width, Height)
dev_set_window_extents (-1, -1, Width, Height)
dev_clear_window ()
dev_display (MosaicImage)
disp_message (WindowHandle, 'Mosaic image', 'window', 12, 12, 'black', 'true')
```

opencv复现：

```opencv
import cv2
import numpy as np

# 1. 读取图像
img1 = cv2.imread('building_01.png')  # 替换为你的路径
img2 = cv2.imread('building_02.png')

# 2. 灰度处理
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 3. 特征提取（ORB 或 SIFT）
orb = cv2.ORB_create(5000)  # 也可用 SIFT_create()
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# 4. 特征匹配（用 Hamming 距离）
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 5. 提取匹配点
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# 6. 使用 RANSAC 估计单应矩阵（projective transform）
H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

# 7. 透视变换并拼接图像
height1, width1 = img1.shape[:2]
height2, width2 = img2.shape[:2]

# 计算输出图像大小（宽度取两者最大值 + 变换宽度）
result = cv2.warpPerspective(img2, H, (width1 + width2, height1))
result[0:height1, 0:width1] = img1  # 把图1放入结果图

# 8. 可视化结果
cv2.imshow('Mosaic Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```



![building_01](F:\常州易管\building_02.png)



![building_01](F:\常州易管\building_01.png)

拼接后：



![pj](F:\常州易管\pj.png)



# NO.07条形码检测

自动检测并解码图像中的条形码，通过绘制条形码的边界框和显示条形码的信息来可视化解码结果

halcon代码

```
* Example program for the usage of the bar code
* reader autodiscrimination feature of HALCON.
* 
* Create bar code reader model
create_bar_code_model ([], [], BarCodeHandle)
set_bar_code_param (BarCodeHandle, 'quiet_zone', 'true')
* 
* Initialization
dev_update_off ()
dev_close_window ()
dev_open_window (0, 0, 512, 512, 'black', WindowHandle)
set_display_font (WindowHandle, 14, 'mono', 'true', 'false')
dev_set_draw ('margin')
dev_set_line_width (3)
dev_set_color ('forest green')
* 
* Define the example images to use
*  for part I
ExampleImagesAny := ['barcode/ean13/ean1301', 'barcode/code39/code3901', 'barcode/gs1databar_omnidir/gs1databar_omnidir_01']
ExpectedNumCodes := [1, 2, 1]
ExpectedData := ['4008535118906', '92274A', 'R74-2001-010', '(01)00098431358722']
*  for part II
ExampleImagesMixed := ['barcode/mixed/barcode_mixed_04', 'barcode/mixed/barcode_mixed_03', 'barcode/mixed/barcode_mixed_01']
* 
* PART I
* 
* Use autodiscrimination to decode any of the bar code types
* supported by HALCON (except PharmaCode) or determine the bar
* code type of unknown bar codes
* 
for IdxExample := 0 to 1 by 1
    * Use different parameters for each run
    * First:  Search for all bar code types
    * Second: Search only expected bar code types
    Message := 'Usage of autodiscrimination to decode'
    Message[1] := 'unknown bar code types.'
    Message[2] := ' '
    if (IdxExample == 0)
        CodeTypes := 'auto'
        Message[3] := 'Part I:'
        Message[4] := 'Compare performance between automatic'
        Message[5] := 'bar code discrimination and search for'
        Message[6] := 'specific code types only.'
        Message[7] := ' '
        Message[8] := 'In the first run we use CodeType=\'' + CodeTypes + '\' to look'
        Message[9] := 'for all bar code types known to HALCON'
        Message[10] := ' '
        Message[11] := 'Please note that'
        Message[12] := ' - Wrong bar codes may be found'
        Message[13] := ' - Execution times are longer than necessary'
        Color := [gen_tuple_const(|Message| - 2,'black'),gen_tuple_const(3,'red')]
    else
        CodeTypes := ['EAN-13', 'Code 39', 'GS1 DataBar Omnidir']
        Message := 'In the second run we look for the expected types only:'
        Message[1] := sum('  ' + CodeTypes)
        Message[2] := 'This reduces the decoding time and'
        Message[3] := 'increases the decoding reliability.'
        Color := gen_tuple_const(|Message|,'black')
    endif
    dev_clear_window ()
    disp_message (WindowHandle, Message, 'window', 12, 12, Color, 'true')
    disp_continue_message (WindowHandle, 'black', 'true')
    stop ()
    * Run the test 2 times with different parameters
    for IdxIma := 0 to |ExampleImagesAny| - 1 by 1
        FileName := ExampleImagesAny[IdxIma]
        read_image (Image, FileName)
        * 
        * Set display defaults
        get_image_size (Image, Width, Height)
        dev_set_window_extents (-1, -1, Width, Height)
        dev_display (Image)
        disp_message (WindowHandle, ['Looking for bar code(s) of type:',sum(' ' + CodeTypes)], 'window', 12, 12, 'black', 'true')
        * 
        * Find and decode bar codes. Measure the time needed.
        count_seconds (Start)
        find_bar_code (Image, SymbolRegions, BarCodeHandle, CodeTypes, DecodedDataStrings)
        count_seconds (Stop)
        Duration := (Stop - Start) * 1000
        * 
        * Check, if results are correct
        Start := sum(ExpectedNumCodes[0:IdxIma]) - ExpectedNumCodes[IdxIma]
        DataExp := ExpectedData[Start:Start + ExpectedNumCodes[IdxIma] - 1]
        WrongIndex := []
        for I := 0 to |DecodedDataStrings| - 1 by 1
            tuple_find (DataExp, DecodedDataStrings[I], Indices)
            if (Indices == -1)
                WrongIndex := [WrongIndex,I]
            endif
        endfor
        Color := ['black',gen_tuple_const(|DecodedDataStrings|,'forest green')]
        Color[WrongIndex + 1] := 'red'
        * 
        * Display results and the execution time
        for I := 1 to |DecodedDataStrings| by 1
            select_obj (SymbolRegions, ObjectSelected, I)
            dev_set_color (Color[I])
            dev_display (ObjectSelected)
        endfor
        get_bar_code_result (BarCodeHandle, 'all', 'decoded_types', DecodedDataTypes)
        disp_message (WindowHandle, ['Found bar code(s) in ' + Duration$'.0f' + ' ms:',DecodedDataTypes + ': ' + DecodedDataStrings], 'window', 5 * 12, 12, Color, 'true')
        disp_continue_message (WindowHandle, 'black', 'true')
        stop ()
    endfor
endfor
* 
* Part II
* 
* Now we decode multiple bar code types
* 
* Display information for the user
Message := 'Part II'
Message[1] := ' '
Message[2] := 'Now we demonstrate the usage of autodiscrimination'
Message[3] := 'to decode multiple mixed bar code types within one image.'
dev_clear_window ()
disp_message (WindowHandle, Message, 'window', 12, 12, 'black', 'true')
disp_continue_message (WindowHandle, 'black', 'true')
stop ()
* 
for IdxExample := 0 to 1 by 1
    if (IdxExample)
        * 
        * We could scan for all bar codes except certain bar code
        * families...
        get_param_info ('find_bar_code', 'CodeType', 'value_list', AllCodeTypes)
        NoGS1 := '~' + regexp_select(AllCodeTypes,'GS1.*')
        NoUPC := '~' + regexp_select(AllCodeTypes,'UPC.*')
        CodeTypes := ['auto',NoGS1,NoUPC]
        CodeTypesDescription := 'all types, except GS1 and UPC variants'
    else
        * 
        * ...or (as we do here) scan only for the EAN family and
        * Code 39
        get_param_info ('find_bar_code', 'CodeType', 'value_list', AllCodeTypes)
        AllEAN := regexp_select(AllCodeTypes,'EAN-13.*')
        CodeTypes := [AllEAN,'Code 39']
        CodeTypesDescription := 'Code 39 and all EAN variants'
    endif
    * 
    * Demonstrate autodiscrimination for mixed bar code types
    for IdxIma := 0 to |ExampleImagesMixed| - 1 by 1
        FileName := ExampleImagesMixed[IdxIma]
        read_image (Image, FileName)
        * 
        * Display image and description
        get_image_size (Image, Width, Height)
        dev_set_window_extents (-1, -1, Width / 2, Height / 2)
        dev_display (Image)
        disp_message (WindowHandle, ['Looking for bar code(s) of type:',' ' + CodeTypesDescription], 'window', 12, 12, 'black', 'true')
        * 
        * Decode mixed bar codes
        find_bar_code (Image, SymbolRegions, BarCodeHandle, CodeTypes, DecodedDataStrings)
        * 
        * Display decoded data and symbol region
        dev_set_color ('forest green')
        dev_display (SymbolRegions)
        get_bar_code_result (BarCodeHandle, 'all', 'decoded_types', DecodedDataTypes)
        area_center (SymbolRegions, Area, Rows, Columns)
        disp_message (WindowHandle, DecodedDataTypes + ': ' + DecodedDataStrings, 'image', Rows, Columns - 160, 'forest green', 'true')
        if (IdxIma < |ExampleImagesMixed| - 1 or IdxExample == 0)
            disp_continue_message (WindowHandle, 'black', 'true')
            stop ()
        endif
    endfor
endfor
```

opencv复现

```python
import cv2
from pyzbar.pyzbar import decode, ZBarSymbol
import numpy as np

# 初始化窗口
window_name = "Barcode Detection"

# 示例图片路径
example_images = ['ean13_defect_01.png']

# 遍历每个图像
for img_path in example_images:
    # 读取图像
    image = cv2.imread(img_path)

    # 解码条形码
    barcodes = decode(image, symbols=[ZBarSymbol.EAN13, ZBarSymbol.CODE39])

    # 显示图像并绘制条形码
    for barcode in barcodes:
        # 获取条形码的矩形边界框
        rect_points = barcode.polygon
        if len(rect_points) == 4:
            pts = np.array(rect_points, dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 255, 0), 3)

        # 获取条形码的类型和数据
        barcode_data = barcode.data.decode('utf-8')
        barcode_type = barcode.type

        # 在图像上添加文字标签
        cv2.putText(image, f"{barcode_data} ({barcode_type})", (barcode.rect[0], barcode.rect[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # 显示图像
    cv2.imshow(window_name, image)

    # 等待按键按下来关闭窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

```



![ean13_defect_01](F:\常州易管\ean13_defect_01.png)

![txm](F:\常州易管\txm.png)



# NO.08 按钮缺陷检测

通过阈值分割和连通域分析，提取按钮的轮廓，并根据面积判断按钮是否成功按下，结果会标记为“OK”或“NG”，并在图像中显示出来。

halcon代码：

```halcon
* 1. 读取图片
* read_image(Image, 'C:/Users/王泽琨/Desktop/MVS_save_anniu/Image_20250717144654798.bmp')

read_image(Image, 'C:/Users/王泽琨/Desktop/MVS_save_anniu/Image_20250717144613655.bmp')
* 2. 分解通道
decompose3(Image, R, G, B)

* 3. 阈值分割橙色区域
threshold(R, RegionR, 240, 255)
threshold(G, RegionG, 110, 220)
threshold(B, RegionB, 10, 70)

intersection(RegionR, RegionG, Temp1)
intersection(Temp1, RegionB, OrangeButton)
dev_display(OrangeButton)

* 4. 连通区域提取（找出所有橘色候选）
connection(OrangeButton, ConnectedRegions)

* 5. 只保留最大面积的橘色区域（主按钮）
select_shape(ConnectedRegions, ButtonRegion, 'area', 'and', 500, 999999)

* 6. 计算面积和重心
area_center(ButtonRegion, Area, Row, Column)

* 7. 显示原图和检测结果
dev_display(Image)
dev_set_color('red')
dev_display(ButtonRegion)
dev_set_color('green')
dev_disp_text('Area: ' + Area, 'image', 10, 10, 'black', [], [])

* 8. NG判定
if (Area < 4700)
    dev_disp_text('OK', 'window', 50, 50, 'red', 'box', 'true')
else
    dev_disp_text('NG', 'window', 50, 50, 'green', 'box', 'true')
endif

```



用opencv复现：

```python
import os
import glob
import cv2

def main():

    image_dir = "/mnt/f/项目文件/anniu_kz0718/10000"
    # 1. 检查目录是否存在
    if not os.path.isdir(image_dir):
        print(f" 目录不存在：{image_dir}")
        return

    # 2. 匹配 .bmp 和 .BMP 文件
    patterns = ["*.bmp"]
    image_paths = []
    for pat in patterns:
        found = glob.glob(os.path.join(image_dir, pat))
        print(f"Pattern {pat!r}: 找到 {len(found)} 个文件")
        image_paths.extend(found)
    image_paths = sorted(set(image_paths))



    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", 1280, 720)

    # 4. 逐张处理
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        # 分离 BGR 通道
        b, g, r = cv2.split(img)

        # 阈值分割橙色区域（R:[200,255], G:[100,210], B:[20,60]）
        mr = cv2.inRange(r, 200, 255)
        mg = cv2.inRange(g, 100, 210)
        mb = cv2.inRange(b, 20, 60)
        mask = cv2.bitwise_and(cv2.bitwise_and(mr, mg), mb)
        # 连通域提取
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 筛选面积
        valid = [(c, cv2.contourArea(c)) for c in cnts
                 if 4000 <= cv2.contourArea(c) <= 12000]

        disp = img.copy()
        if not valid:
            cv2.putText(disp, "No Button Found", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        else:
            # 取最大面积区域
            c, area = max(valid, key=lambda x: x[1])
            M = cv2.moments(c)
            cx = int(M["m10"]/M["m00"]) if M["m00"] else 0
            cy = int(M["m01"]/M["m00"]) if M["m00"] else 0

            cv2.drawContours(disp, [c], -1, (0,0,255), 2)
            cv2.circle(disp, (cx, cy), 5, (0,255,0), -1)
            cv2.putText(disp, f"Area: {int(area)}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

            label = "OK" if area < 3700 else "NG"
            color = (0,0,255) if label == "OK" else (0,255,0)
            cv2.putText(disp, label, (50,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, color, 3)

        print(label)
        cv2.imshow("Result", disp)
        key = cv2.waitKey(0)
        if key == 27:  # 按 Esc 键退出
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

```



![anniu](F:\常州易管\anniu.png)