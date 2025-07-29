import cv2
import numpy as np
import matplotlib.pyplot as plt


# 初始化
def display_image(image, window_name='Image'):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 创建高斯滤波器
def create_gaussian_filter(sigma, size):
    return cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T


# 快速傅里叶变换 (FFT)
def fft_convolve(image, filter):
    # 转换为频域
    f_image = np.fft.fft2(image)
    f_filter = np.fft.fft2(filter, s=image.shape)

    # 卷积 (频域乘积)
    f_convol = f_image * f_filter

    # 逆变换
    convol_image = np.fft.ifft2(f_convol)

    return np.abs(convol_image)


# 处理图像
def process_image(image, filter):
    # 将图像转换为灰度
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 进行 FFT 卷积
    convol_image = fft_convolve(gray_image, filter)

    # 对卷积结果进行处理（如灰度范围和阈值）
    _, threshold_image = cv2.threshold(convol_image, np.max(convol_image) * 0.8, 255, cv2.THRESH_BINARY)

    # 找到连接区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(threshold_image))

    # 绘制检测到的缺陷
    result_image = image.copy()
    for i in range(1, num_labels):  # 跳过背景
        if stats[i, cv2.CC_STAT_AREA] > 10:  # 过滤小区域
            x, y, w, h, area = stats[i]
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # 在区域中心画圆
            cv2.circle(result_image, (int(centroids[i][0]), int(centroids[i][1])), 10, (0, 255, 0), 2)

    # 显示结果
    if num_labels > 1:
        result_message = f"Not OK, {num_labels - 1} defect(s) found"
        color = (0, 0, 255)  # Red for defects
    else:
        result_message = 'OK'
        color = (0, 255, 0)  # Green for no defects

    cv2.putText(result_image, result_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    display_image(result_image)


# 主程序
def main():
    # 读取图像
    num_images = 11
    for index in range(1, num_images + 1):
        image_path = f"plastics_{index:02d}.png"
        image = cv2.imread(image_path)

        # 创建高斯滤波器
        sigma1, sigma2 = 10.0, 3.0
        filter1 = create_gaussian_filter(sigma1, image.shape[0])
        filter2 = create_gaussian_filter(sigma2, image.shape[0])
        filter = filter1 - filter2

        # 处理图像
        process_image(image, filter)


if __name__ == '__main__':
    main()

"""
表面检测 detect_indent_fft

* This program demonstrates how to detect small texture
* defects on the surface of plastic items by using the fast
* Fourier transform (FFT).
* First, we construct a suitable filter using Gaussian
* filters. Then, the images and the filter are convolved
* by using fast Fourier transforms. Finally, the defects
* are detected in the filtered images by using
* morphology operators.
* 
* Initializations
dev_update_off ()
dev_close_window ()
read_image (Image, 'plastics/plastics_01')
get_image_size (Image, Width, Height)
dev_open_window (0, 0, Width, Height, 'black', WindowHandle)
set_display_font (WindowHandle, 14, 'mono', 'true', 'false')
dev_set_draw ('margin')
dev_set_line_width (3)
dev_set_color ('red')
* 
* Optimize the fft speed for the specific image size
optimize_rft_speed (Width, Height, 'standard')
* 
* Construct a suitable filter by combining two Gaussian
* filters
Sigma1 := 10.0
Sigma2 := 3.0
gen_gauss_filter (GaussFilter1, Sigma1, Sigma1, 0.0, 'none', 'rft', Width, Height)
gen_gauss_filter (GaussFilter2, Sigma2, Sigma2, 0.0, 'none', 'rft', Width, Height)
sub_image (GaussFilter1, GaussFilter2, Filter, 1, 0)
* 
* Process the images iteratively
NumImages := 11
for Index := 1 to NumImages by 1
    * 
    * Read an image and convert it to gray values
    read_image (Image, 'plastics/plastics_' + Index$'02')
    rgb1_to_gray (Image, Image)
    * Perform the convolution in the frequency domain
    rft_generic (Image, ImageFFT, 'to_freq', 'none', 'complex', Width)
    convol_fft (ImageFFT, Filter, ImageConvol)
    rft_generic (ImageConvol, ImageFiltered, 'from_freq', 'n', 'real', Width)
    * 
    * Process the filtered image
    gray_range_rect (ImageFiltered, ImageResult, 10, 10)
    min_max_gray (ImageResult, ImageResult, 0, Min, Max, Range)
    threshold (ImageResult, RegionDynThresh, max([5.55,Max * 0.8]), 255)
    connection (RegionDynThresh, ConnectedRegions)
    select_shape (ConnectedRegions, SelectedRegions, 'area', 'and', 4, 99999)
    union1 (SelectedRegions, RegionUnion)
    closing_circle (RegionUnion, RegionClosing, 10)
    connection (RegionClosing, ConnectedRegions1)
    select_shape (ConnectedRegions1, SelectedRegions1, 'area', 'and', 10, 99999)
    area_center (SelectedRegions1, Area, Row, Column)
    * 
    * Display the results
    dev_display (Image)
    Number := |Area|
    if (Number)
        gen_circle_contour_xld (ContCircle, Row, Column, gen_tuple_const(Number,30), gen_tuple_const(Number,0), gen_tuple_const(Number,rad(360)), 'positive', 1)
        ResultMessage := ['Not OK',Number + ' defect(s) found']
        Color := ['red', 'black']
        dev_display (ContCircle)
    else
        ResultMessage := 'OK'
        Color := 'forest green'
    endif
    disp_message (WindowHandle, ResultMessage, 'window', 12, 12, Color, 'true')
    if (Index != NumImages)
        disp_continue_message (WindowHandle, 'black', 'true')
        stop ()
    endif
endfor
"""