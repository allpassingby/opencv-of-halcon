import os
import glob
import cv2

def main():

    image_dir = "10000"
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
