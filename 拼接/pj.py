import cv2
import numpy as np

# 1. 读取图像
img1 = cv2.imread('pcb_01.png')  # 替换为你的路径
img2 = cv2.imread('pcb_02.png')

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
result = cv2.warpPerspective(img2, H, (width1 + width2, height1+height2))
result[0:height1, 0:width1] = img1  # 把图1放入结果图

# 8. 可视化结果
cv2.imshow('Mosaic Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""halcon代码
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

"""
