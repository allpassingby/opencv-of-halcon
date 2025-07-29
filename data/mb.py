import cv2
import numpy as np

def rotate_image(image, angle):
    """绕中心旋转并扩边避免裁剪（背景黑）"""
    h, w = image.shape[:2]
    M    = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nw = int(h*sin + w*cos)
    nh = int(h*cos + w*sin)
    M[0,2] += (nw - w)/2
    M[1,2] += (nh - h)/2
    return cv2.warpAffine(image, M, (nw, nh), borderValue=0)

def refine_match(scene_gray, tpl, init_angle, init_loc, search_offset=10):
    """以粗匹配结果为中心做 ±5° 细调"""
    best = {
        'score':   -1,
        'angle':   init_angle,
        'loc':     init_loc,
        'rt':      None,
        'edge':    None,
        'shape':   (0,0),
    }
    x0, y0 = init_loc
    for da in np.arange(-5, 5.1, 1):
        angle = init_angle + da
        rt    = rotate_image(tpl, angle)
        rt_edge = cv2.Canny(rt, 50, 150)
        h, w = rt.shape
        ys = max(0, y0 - search_offset)
        xs = max(0, x0 - search_offset)
        sub = scene_gray[ys:ys+h+2*search_offset, xs:xs+w+2*search_offset]
        if sub.shape[0] < h or sub.shape[1] < w:
            continue
        res = cv2.matchTemplate(
            sub, rt,
            cv2.TM_CCORR_NORMED,
            mask=(rt_edge>0).astype(np.uint8)
        )
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > best['score']:
            best.update({
                'score': score,
                'angle': angle,
                'loc':   (loc[0] + xs, loc[1] + ys),
                'rt':    rt,
                'edge':  rt_edge,
                'shape': (h, w),
            })
    return best

def main():
    # 1. 读图 & 预处理
    scene      = cv2.imread('taijie.jpg')
    scene_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    tpl        = cv2.imread('taijiemob.jpg', cv2.IMREAD_GRAYSCALE)

    # 2. 粗匹配：收集每个角度的分数
    matches = []
    for angle in range(0, 360, 5):
        rt      = rotate_image(tpl, angle)
        rt_edge = cv2.Canny(rt, 50, 150)
        h, w    = rt.shape
        if h > scene_gray.shape[0] or w > scene_gray.shape[1]:
            continue
        res = cv2.matchTemplate(
            scene_gray, rt,
            cv2.TM_CCORR_NORMED,
            mask=(rt_edge>0).astype(np.uint8)
        )
        _, score, _, loc = cv2.minMaxLoc(res)
        matches.append((score, angle, loc, rt, rt_edge))

    if len(matches) < 1:
        print("未找到任何匹配")
        return

    # 3. 排序 + 取前2 + 得分过滤
    matches.sort(key=lambda x: x[0], reverse=True)
    # 最优
    score1, angle1, loc1, rt1, edge1 = matches[0]
    top2 = [(score1, angle1, loc1)]
    # 次优（仅当满足条件才加入）
    if len(matches) > 1:
        score2, angle2, loc2, rt2, edge2 = matches[1]
        if score2 >= max(0.2, score1 * 0.6):
            top2.append((score2, angle2, loc2))

    # 4. 细调 & 可视化
    out    = scene.copy()
    colors = [(0,255,0), (255,0,0)]
    for idx, (scr, ang, loc) in enumerate(top2):
        refined = refine_match(scene_gray, tpl, ang, loc)
        x, y    = refined['loc']
        h, w    = refined['shape']
        color   = colors[idx]
        # 画精确轮廓
        cnts,_ = cv2.findContours(
            (refined['edge']>0).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        for cnt in cnts:
            cnt2 = cnt + np.array([[[x,y]]])
            cv2.drawContours(out, [cnt2], -1, color, 2)
        # 画外接矩形
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 1)
        # 标注
        cv2.putText(
            out,
            f"{refined['angle']:.1f}° {refined['score']:.2f}",
            (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

    cv2.namedWindow('Filtered Matches', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Filtered Matches', 900, 600)
    cv2.imshow('Filtered Matches', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
