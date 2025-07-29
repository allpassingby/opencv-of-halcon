import sys
import csv
import cv2
import numpy as np
from pathlib import Path
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt

class PolygonLabel(QtWidgets.QLabel):
    """
    支持缩放、平移和多边形点添加的自定义 QLabel。
    """
    polygon_finished = QtCore.pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.drawing = False
        self.selected_point = -1
        self.zoom = 1.0            # 缩放比例
        self._pixmap = None        # 原始 QPixmap
        self.image_rect = None     # 图像显示区域
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet("background-color: #2D2D30;")  # 深灰色背景

        # 提高框选精准度
        self.point_radius = 8  # 点半径
        self.point_hit_range = 12  # 点检测范围

    def setPixmap(self, pixmap: QtGui.QPixmap):
        # 保存原始 pixmap 并应用当前缩放
        self._pixmap = pixmap
        self.apply_zoom()

    def apply_zoom(self):
        if self._pixmap is None:
            return
        # 根据缩放比例调整 pixmap 大小
        w = int(self._pixmap.width() * self.zoom)
        h = int(self._pixmap.height() * self.zoom)
        scaled = self._pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        super().setPixmap(scaled)
        # 计算图像显示区域（考虑对齐）
        self.image_rect = QtCore.QRect(
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2,
            scaled.width(),
            scaled.height()
        )

    def mapToImage(self, pos):
        """将窗口坐标映射到原始图像坐标"""
        if self._pixmap is None or self.image_rect is None:
            return None

        # 检查点击是否在图像区域内
        if not self.image_rect.contains(pos):
            return None

        # 转换为相对于图像显示区域的坐标
        x_in_image = (pos.x() - self.image_rect.x()) / self.zoom
        y_in_image = (pos.y() - self.image_rect.y()) / self.zoom
        return QtCore.QPoint(int(x_in_image), int(y_in_image))

    def mapToScreen(self, pt):
        """将图像坐标映射到屏幕坐标"""
        if self.image_rect is None:
            return None
        return QtCore.QPoint(
            self.image_rect.x() + int(pt.x() * self.zoom),
            self.image_rect.y() + int(pt.y() * self.zoom)
        )

    def wheelEvent(self, event: QtGui.QWheelEvent):
        # 鼠标滚轮触发缩放
        delta = event.angleDelta().y()
        factor = 1.25 if delta > 0 else 0.8
        self.zoom = max(0.1, min(self.zoom * factor, 10.0))
        self.apply_zoom()
        event.accept()

    def mousePressEvent(self, event):
        # 坐标映射：将屏幕坐标映射到原始图像坐标
        if self._pixmap is None:
            return

        img_pt = self.mapToImage(event.pos())
        if img_pt is None:  # 点击在图像区域外
            return

        if event.button() == QtCore.Qt.LeftButton:
            # 左键：添加或选中点
            for i, pt in enumerate(self.points):
                # 使用欧几里得距离提高精准度
                distance = ((img_pt.x() - pt.x())**2 + (img_pt.y() - pt.y())**2)**0.5
                if distance < self.point_hit_range:
                    self.selected_point = i
                    return
            self.drawing = True
            self.points.append(img_pt)
            self.selected_point = -1
            self.update()
        elif event.button() == QtCore.Qt.RightButton:
            # 右键：删除点或完成多边形
            for i, pt in enumerate(self.points):
                distance = ((img_pt.x() - pt.x())**2 + (img_pt.y() - pt.y())**2)**0.5
                if distance < self.point_hit_range:
                    del self.points[i]
                    self.selected_point = -1
                    self.update()
                    return
            if self.drawing and len(self.points) >= 3:
                self.polygon_finished.emit([(p.x(), p.y()) for p in self.points])
                self.drawing = False
                self.update()

    def mouseDoubleClickEvent(self, event):
        """双击添加点"""
        if self._pixmap is None:
            return
        img_pt = self.mapToImage(event.pos())
        if img_pt and event.button() == QtCore.Qt.LeftButton:
            self.points.append(img_pt)
            self.update()

    def keyPressEvent(self, event):
        # Delete/Backspace 删除选中点，Esc 取消选中
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if 0 <= self.selected_point < len(self.points):
                del self.points[self.selected_point]
                self.selected_point = -1
                self.update()
            elif self.points:
                self.points.pop()
                self.update()
        elif event.key() == Qt.Key_Escape:
            self.selected_point = -1
            self.update()
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event):
        # 拖拽移动选中点
        if 0 <= self.selected_point < len(self.points):
            img_pt = self.mapToImage(event.pos())
            if img_pt:
                self.points[self.selected_point] = img_pt
                self.update()
        super().mouseMoveEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.points:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)  # 启用抗锯齿
        painter.setFont(QtGui.QFont("Arial", 10))

        # 绘制顶点
        for i, pt in enumerate(self.points):
            screen_pt = self.mapToScreen(pt)
            if not screen_pt:
                continue

            # 设置画笔 - 使用不同颜色区分选中点
            if i == self.selected_point:
                pen = QtGui.QPen(QtCore.Qt.red)
                brush = QtGui.QBrush(QtCore.Qt.red)
            else:
                pen = QtGui.QPen(QtCore.Qt.cyan)
                brush = QtGui.QBrush(QtCore.Qt.cyan)

            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(brush)

            # 绘制点
            painter.drawEllipse(screen_pt, self.point_radius, self.point_radius)

            # 绘制标签
            painter.setPen(QtGui.QPen(QtCore.Qt.yellow))
            painter.drawText(screen_pt.x() + 10, screen_pt.y() - 8, f"P{i+1}:({pt.x()},{pt.y()})")

        # 绘制多边形线 - 使用蓝色
        if len(self.points) > 1:
            pen = QtGui.QPen(QtGui.QColor(0, 200, 255))  # 亮蓝色
            pen.setWidth(2)
            painter.setPen(pen)
            path = QtGui.QPainterPath()

            # 起点
            start_pt = self.mapToScreen(self.points[0])
            if start_pt:
                path.moveTo(start_pt)

            # 连接点
            for pt in self.points[1:]:
                p = self.mapToScreen(pt)
                if p:
                    path.lineTo(p)

            # 闭合多边形
            if not self.drawing:
                if start_pt:
                    path.lineTo(start_pt)

            painter.drawPath(path)

        # 绘制操作提示
        painter.setPen(QtGui.QPen(QtCore.Qt.white))
        painter.drawText(10, 20, "左键:添加/选中  右键:删除/完成  Delete:删除点  Esc:取消选中  滚轮:缩放")
        painter.end()

class TemplateMaker(QtWidgets.QWidget):
    """
    主窗口：包含滚动区域显示 PolygonLabel 及功能按钮。
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高精度多边形标注工具")
        self.setMinimumSize(800, 600)

        # 图像与掩码等数据
        self.image = None
        self.mask = None
        self.last_vis = None
        self.orig_pts = []

        # 中心对齐的滚动区
        self.label = PolygonLabel()
        self.label.setAlignment(Qt.AlignCenter)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.label)
        scroll.setWidgetResizable(True)
        scroll.setAlignment(Qt.AlignCenter)

        # 功能按钮
        self.btn_load = QtWidgets.QPushButton("导入图像")
        self.btn_pre = QtWidgets.QPushButton("预处理图像")
        self.btn_save = QtWidgets.QPushButton("保存模板")
        self.btn_export = QtWidgets.QPushButton("导出坐标")
        self.btn_clear = QtWidgets.QPushButton("清除所有点")

        # 初始化按钮状态
        for btn in (self.btn_pre, self.btn_save, self.btn_export, self.btn_clear):
            btn.setEnabled(False)

        # 布局
        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(scroll)

        # 按钮行
        hb = QtWidgets.QHBoxLayout()
        hb.addWidget(self.btn_load)
        hb.addWidget(self.btn_pre)
        hb.addWidget(self.btn_save)
        hb.addWidget(self.btn_export)
        hb.addWidget(self.btn_clear)

        vbox.addLayout(hb)
        vbox.setStretch(0, 1)  # 图像区域占主要空间

        # 状态栏
        self.status_bar = QtWidgets.QStatusBar()
        self.status_bar.showMessage("就绪")
        vbox.addWidget(self.status_bar)

        # 按钮样式
        button_style = """
            QPushButton {
                padding: 8px 16px;
                font-weight: bold;
                min-width: 100px;
                border-radius: 4px;
                background-color: #0078D7;
                color: white;
            }
            QPushButton:hover {
                background-color: #106EBE;
            }
            QPushButton:pressed {
                background-color: #005A9E;
            }
            QPushButton:disabled {
                background-color: #3F3F46;
                color: #A0A0A0;
            }
        """
        for btn in [self.btn_load, self.btn_pre, self.btn_save,
                    self.btn_export, self.btn_clear]:
            btn.setStyleSheet(button_style)
            btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # 信号连接
        self.btn_load.clicked.connect(self.load_image)
        self.btn_pre.clicked.connect(self.preprocess_image)
        self.btn_save.clicked.connect(self.save_template)
        self.btn_export.clicked.connect(self.export_coordinates)
        self.btn_clear.clicked.connect(self.clear_points)
        self.label.polygon_finished.connect(self.on_polygon_finished)

    def load_image(self):
        try:
            # 释放之前的内存
            self.image = None
            self.mask = None
            self.last_vis = None

            # 选择图像文件
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "选择图像", "",
                "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
            )
            if not path:
                return

            # 加载图像
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"无法加载图片：{path}")

            self.image = img
            self.last_vis = None
            self.orig_pts.clear()
            self.reset_state()
            self.update_display(self.image)
            self.status_bar.showMessage(f"已加载图像: {Path(path).name} | 尺寸: {img.shape[1]}x{img.shape[0]}")
            self.btn_pre.setEnabled(True)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "加载错误", f"错误: {str(e)}")

    def reset_state(self):
        self.label.points.clear()
        self.label.selected_point = -1
        self.label.zoom = 1.0
        self.mask = None

        # 重置按钮状态
        self.btn_pre.setEnabled(False)
        for btn in (self.btn_save, self.btn_export, self.btn_clear):
            btn.setEnabled(False)

    def clear_points(self):
        self.label.points.clear()
        self.label.selected_point = -1
        self.label.update()
        self.last_vis = None
        self.orig_pts.clear()

        if self.image is not None:
            self.update_display(self.image)

        for btn in (self.btn_save, self.btn_export):
            btn.setEnabled(False)

        self.status_bar.showMessage("已清除所有点")

    def preprocess_image(self):
        try:
            if self.image is None:
                return

            img = self.image.copy()

            # 转换为LAB颜色空间并增强亮度通道
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

            # Gamma校正增强对比度
            inv_gamma = 1.0 / 0.8
            table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], np.uint8)
            img = cv2.LUT(img, table)

            # 应用预处理结果
            self.image = img
            self.clear_points()  # 预处理后清除已有点
            self.update_display(self.image)
            self.status_bar.showMessage("图像预处理完成")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "预处理错误", f"发生错误: {str(e)}")

    def on_polygon_finished(self, poly):
        try:
            if len(poly) < 3:
                QtWidgets.QMessageBox.warning(self, "无效多边形", "至少需要3个点构成多边形")
                return

            self.orig_pts = [(x, y) for x, y in poly]
            h, w = self.image.shape[:2]
            mask = np.zeros((h, w), np.uint8)
            pts = np.array(self.orig_pts, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            self.mask = mask

            overlay = self.image.copy()
            # 用纯 BGR 填充
            cv2.fillPoly(overlay, [pts], (255, 150, 0))

            # 先把原图放在 src1，再把 overlay 放在 src2：
            alpha = 0.3  # overlay 的透明度，取值 0.0~1.0
            beta = 1.0 - alpha
            # 注意顺序：第一个参数是原图，第二个参数是它的权重；第三个参数是 overlay，第四个参数是 overlay 的权重
            self.last_vis = cv2.addWeighted(self.image, beta, overlay, alpha, 0)

            # 最后再刷新显示
            self.update_display(self.last_vis)

            # 更新按钮状态
            self.btn_save.setEnabled(True)
            self.btn_export.setEnabled(True)
            self.btn_clear.setEnabled(True)

            self.status_bar.showMessage(f"多边形已创建，包含 {len(self.orig_pts)} 个点")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "多边形错误", f"创建多边形时出错: {str(e)}")

    def save_template(self):
        try:
            if not self.orig_pts:
                QtWidgets.QMessageBox.warning(self, "提示", "请先创建多边形")
                return

            # 保存模板文件
            tmpl, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "保存模板", "", "NumPy 文件 (*.npz)"
            )
            if tmpl:
                # 保存多边形坐标和图像尺寸
                np.savez_compressed(
                    tmpl,
                    polygon=self.orig_pts,
                    image_size=self.image.shape[:2]
                )

            # 保存图像
            imgf, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "保存图像", "", "PNG (*.png);;JPEG (*.jpg);;TIFF (*.tiff)"
            )
            if imgf:
                cv2.imwrite(imgf, self.last_vis)

            QtWidgets.QMessageBox.information(
                self, "完成", "模板和图像已保存！"
            )
            self.status_bar.showMessage("模板和图像已保存")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "保存错误", f"保存时出错: {str(e)}")

    def export_coordinates(self):
        try:
            if not self.orig_pts:
                QtWidgets.QMessageBox.warning(self, "提示", "无坐标可导出")
                return

            # 选择保存路径
            fn, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "保存坐标", "",
                "CSV (*.csv);;TXT (*.txt);;All Files (*)"
            )
            if not fn:
                return

            # 保存为CSV
            if fn.lower().endswith('.csv'):
                with open(fn, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['序号', 'X坐标', 'Y坐标'])
                    for i, (x, y) in enumerate(self.orig_pts, 1):
                        w.writerow([i, x, y])
            # 保存为TXT
            else:
                with open(fn, 'w', encoding='utf-8') as f:
                    f.write('序号\tX坐标\tY坐标\n')
                    for i, (x, y) in enumerate(self.orig_pts, 1):
                        f.write(f"{i}\t{x}\t{y}\n")

            QtWidgets.QMessageBox.information(
                self, "完成", f"已保存坐标文件: {Path(fn).name}"
            )
            self.status_bar.showMessage(f"已导出坐标到 {Path(fn).name}")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "导出错误", f"导出坐标时出错: {str(e)}")

    def update_display(self, img):
        """将OpenCV图像转换为QPixmap并显示"""
        try:
            # 转换颜色空间
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w

            # 创建QImage并转换为QPixmap
            qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg)

            # 更新显示
            self.label.setPixmap(pix)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "显示错误", f"更新显示时出错: {str(e)}")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    # 设置应用样式
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(45, 45, 48))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(37, 37, 38))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 48))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 48))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(0, 122, 204))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)

    # 创建主窗口
    win = TemplateMaker()
    win.showMaximized()
    sys.exit(app.exec_())