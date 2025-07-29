import cv2
import numpy as np

def rotate_image(image, angle):
    """绕中心旋转并扩边避免裁剪（背景置0）"""
    h, w = image.shape[:2]
    M    = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nw = int(h*sin + w*cos)
    nh = int(h*cos + w*sin)
    M[0,2] += (nw - w)/2
    M[1,2] += (nh - h)/2
    return cv2.warpAffine(image, M, (nw, nh), borderValue=0)

# —— 1. 读图
scene      = cv2.imread('02.jpg')
scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
tpl        = cv2.imread('temple.jpg', cv2.IMREAD_GRAYSCALE)

best = {'score': -1}

# —— 第一级：粗角度搜索 —— #
for angle in range(0, 360, 5):
    rt      = rotate_image(tpl, angle)
    rt_edge = cv2.Canny(rt, 50, 150)
    h, w    = rt.shape
    if h > scene_gray.shape[0] or w > scene_gray.shape[1]:
        continue

    res = cv2.matchTemplate(
        scene_gray, rt,
        cv2.TM_CCORR_NORMED,
        mask=(rt_edge > 0).astype(np.uint8)
    )
    _, score, _, loc = cv2.minMaxLoc(res)
    if score > best['score']:
        best.update(score=score, angle=angle, loc=loc, rt=rt, edge=rt_edge)

# —— 第二级：小窗口细调 —— #
angle0, (x0, y0) = best['angle'], best['loc']
for da in np.arange(-5, 5.1, 1):  # ±5°，步长1°
    angle  = angle0 + da
    rt     = rotate_image(tpl, angle)
    rt_edge= cv2.Canny(rt, 50, 150)
    h, w   = rt.shape
    if h > scene_gray.shape[0] or w > scene_gray.shape[1]:
        continue

    # 在 x0±10, y0±10 区域内搜
    ys, xs = y0 - 10, x0 - 10
    sub = scene_gray[ys:ys+h+20, xs:xs+w+20]
    if sub.shape[0] < h or sub.shape[1] < w:
        continue

    res = cv2.matchTemplate(
        sub, rt,
        cv2.TM_CCORR_NORMED,
        mask=(rt_edge > 0).astype(np.uint8)
    )
    _, score, _, loc = cv2.minMaxLoc(res)
    if score > best['score']:
        best.update(
            score=score,
            angle=angle,
            loc=(loc[0] + xs, loc[1] + ys),
            rt=rt,
            edge=rt_edge
        )

# —— 最终可视化 —— #
x, y    = best['loc']
h, w    = best['rt'].shape
disp    = scene.copy()

# —— 新增：只保留模板对应区域，其他设为黑 —— #
# 1. 二值化旋转后的模板，得到精确掩膜
_, mask_tpl = cv2.threshold(best['rt'], 1, 255, cv2.THRESH_BINARY)
mask_bool   = mask_tpl.astype(bool)

# 2. 在 disp 的匹配框区域内将掩膜外的像素置黑
sub_disp = disp[y:y+h, x:x+w]
sub_disp[~mask_bool] = 0
disp[y:y+h, x:x+w] = sub_disp

# —— 然后再画出模板轮廓 —— #
cnts, _ = cv2.findContours(mask_tpl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for cnt in cnts:
    cnt_shift = cnt + np.array([[x, y]])
    cv2.drawContours(disp, [cnt_shift], -1, (0, 255, 0), 2)

# —— 显示角度与得分 —— #
cv2.putText(
    disp,
    f"Angle: {best['angle']:.1f}°  Score: {best['score']:.3f}",
    (x, y - 15),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6, (0, 255, 0), 2
)

# —— 弹窗展示 —— #
cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Result', 800, 600)
cv2.imshow('Result', disp)
cv2.waitKey(0)
cv2.destroyAllWindows()
