import cv2
import numpy as np

def rotate_image(image, angle):
    """绕中心旋转并扩边"""
    h, w = image.shape[:2]
    M    = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nw = int(h*sin + w*cos)
    nh = int(h*cos + w*sin)
    M[0,2] += (nw - w)/2
    M[1,2] += (nh - h)/2
    return cv2.warpAffine(image, M, (nw, nh), borderValue=0)

scene = cv2.imread('02.jpg')
tpl   = cv2.imread('temple.jpg', cv2.IMREAD_GRAYSCALE)
scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
tpl_edge   = cv2.Canny(tpl,  50, 150)

best = {'score':-1}

# —— 第一级：粗角度搜索 —— #
for angle in range(0, 360, 5):
    rt     = rotate_image(tpl, angle)
    rt_edge= cv2.Canny(rt,50,150)
    h, w   = rt.shape
    if h>scene.shape[0] or w>scene.shape[1]: continue

    res = cv2.matchTemplate(
        scene_gray, rt,
        cv2.TM_CCORR_NORMED,
        mask=(rt_edge>0).astype(np.uint8)
    )
    _, score, _, loc = cv2.minMaxLoc(res)
    if score > best['score']:
        best.update(score=score, angle=angle, loc=loc, rt=rt, edge=rt_edge)

# —— 第二级：小窗口细调 —— #
angle0, (x0,y0) = best['angle'], best['loc']
for da in np.arange(-5, 5.1, 1):               # ±5°，步长1°
    angle = angle0 + da
    rt     = rotate_image(tpl, angle)
    rt_edge= cv2.Canny(rt,50,150)
    h, w   = rt.shape
    if h>scene.shape[0] or w>scene.shape[1]: continue

    # 在 x0±10, y0±10 区域内搜
    sub = scene_gray[y0-10:y0+10+h, x0-10:x0+10+w]
    if sub.shape[0]<h or sub.shape[1]<w: continue

    res = cv2.matchTemplate(
        sub, rt,
        cv2.TM_CCORR_NORMED,
        mask=(rt_edge>0).astype(np.uint8)
    )
    _, score, _, loc = cv2.minMaxLoc(res)
    if score > best['score']:
        # loc 是相对于 sub 的
        best.update(
            score=score, angle=angle,
            loc=(loc[0]+x0-10, loc[1]+y0-10),
            rt=rt, edge=rt_edge
        )

# —— 最终可视化 —— #
x, y    = best['loc']
h, w    = best['rt'].shape
disp    = scene.copy()

# 从匹配边缘中提取真正的轮廓
cnts,_ = cv2.findContours(best['edge'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for cnt in cnts:
    # 平移到 scene 坐标系
    cnt_shift = cnt + np.array([[x,y]])
    cv2.drawContours(disp, [cnt_shift], -1, (0,255,0), 2)

cv2.putText(
    disp,
    f"Angle: {best['angle']:.1f}°  Score: {best['score']:.3f}",
    (x, y-15),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6, (0,255,0), 2
)

cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Result', 800, 600)
cv2.imshow('Result', disp)
cv2.waitKey(0)
cv2.destroyAllWindows()
