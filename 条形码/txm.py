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
