# encoding:utf8
# 打包
# env: pyinstaller==4.6，opencv-python==4.6
# run: pyinstaller -F -w -n TensorRT模型实时监控 demo_qt.py

import sys

import qimage2ndarray
import win32gui
from mmdeploy_python import Detector
import cv2
import numpy as np
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QFont, QPixmap, QImage, QDesktopServices
from PyQt5.QtWidgets import QGridLayout, QLabel, QPushButton, QFileDialog, QWidget, QApplication

hwnd = win32gui.FindWindow(None, 'C:\Windows\system32\cmd.exe')
coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear',
                'hair drier', 'toothbrush']

button_css = 'QPushButton{color:white;background-color:rgb(14 , 150 , 254);border-radius:2px;}QPushButton:hover{color:white;background-color:rgb(44 , 137 , 255);}QPushButton:pressed{color:white;background-color:rgb(14 , 135 , 228);padding-left:1px;padding-top:1px;}'


class Main(QWidget):

    def __init__(self):
        super().__init__()
        self.app_size = [1422, 800]
        self.setMinimumSize(self.app_size[0], self.app_size[1])
        self.setWindowTitle('TensorRT模型实时监控')

        self.detector = None  # Detector
        self.window = QLabel(self)  # 推理结果窗口
        self.bg = QPixmap('bg.jpg')  # 推理结构窗口背景图片
        self.window_size = [self.bg.width(), self.bg.height()]
        self.visiual_thresh = 0.5
        self.monitor_tag = False

        self.author_info = {'blog': 'https://blog.csdn.net/qq_39435411',
                            'blog1': 'https://www.cnblogs.com/gy77/',
                            'github': 'https://github.com/gy-7'}

        self.initUI()  # 初始化UI界面
        # self.setStyleSheet("background-color: rgb(255, 255, 255);") # 设置底色

        self.desktop = QApplication.desktop()
        self.desktop_width, self.desktop_height = int(self.desktop.width()), int(self.desktop.height())

    def initUI(self):
        self.grid = QGridLayout()
        self.setLayout(self.grid)

        self.grid.addWidget(self.window, 0, 0, 20, 20)
        self.window.setPixmap(self.bg)

        # 初始化按钮
        self.init_control_control()

    def init_control_control(self):
        self.info = QLabel(' 模型加载状态 : ')
        self.status = QLabel(' 模型未加载 ')
        self.load_model = QPushButton('加载模型', self)
        self.inf_img = QPushButton('推理图片', self)
        self.monitor_desktop = QPushButton('监控桌面', self)
        self.monitor_off = QPushButton('关闭监控', self)
        self.author_info_label = QLabel('作者信息: ')
        self.author_name = QPushButton('gy77', self)
        self.author_blog = QPushButton('CSDN', self)
        self.author_blog1 = QPushButton('博客园', self)
        self.author_github = QPushButton('GitHub', self)

        self.info.setFont(QFont("微软雅黑", 16))
        self.info.setStyleSheet("background-color: rgb(106, 137, 204); color: white;")
        self.status.setFont(QFont("微软雅黑", 16))
        self.status.setStyleSheet("background-color: rgb(254, 202, 87);")
        self.load_model.setFont(QFont("微软雅黑", 16))
        self.inf_img.setFont(QFont("微软雅黑", 16))
        self.monitor_desktop.setFont(QFont("微软雅黑", 16))
        self.monitor_off.setFont(QFont("微软雅黑", 16))
        self.author_info_label.setFont(QFont("微软雅黑", 16))
        self.author_name.setFont(QFont("微软雅黑", 16))
        self.author_name.setStyleSheet(button_css)
        self.author_blog.setFont(QFont("微软雅黑", 16))
        self.author_blog.setStyleSheet(button_css)
        self.author_blog1.setFont(QFont("微软雅黑", 16))
        self.author_blog1.setStyleSheet(button_css)
        self.author_github.setFont(QFont("微软雅黑", 16))
        self.author_github.setStyleSheet(button_css)

        self.info.resize(self.info.sizeHint())
        self.status.resize(self.info.sizeHint())
        self.load_model.resize(self.load_model.sizeHint())
        self.inf_img.resize(self.inf_img.sizeHint())
        self.monitor_desktop.resize(self.monitor_desktop.sizeHint())
        self.monitor_off.resize(self.monitor_off.sizeHint())
        self.author_info_label.resize(self.author_info_label.sizeHint())
        self.author_name.resize(self.author_name.sizeHint())
        self.author_blog.resize(self.author_blog.sizeHint())
        self.author_blog1.resize(self.author_blog1.sizeHint())
        self.author_github.resize(self.author_github.sizeHint())

        self.grid.addWidget(self.info, 0, 20, 1, 2)
        self.grid.addWidget(self.status, 1, 20, 1, 2)
        self.grid.addWidget(self.load_model, 3, 20, 1, 2)
        self.grid.addWidget(self.inf_img, 4, 20, 1, 2)
        self.grid.addWidget(self.monitor_desktop, 5, 20, 1, 2)
        self.grid.addWidget(self.monitor_off, 6, 20, 1, 2)
        self.grid.addWidget(self.author_info_label, 15, 20, 1, 2)
        self.grid.addWidget(self.author_name, 16, 20, 1, 2)
        self.grid.addWidget(self.author_blog, 17, 20, 1, 2)
        self.grid.addWidget(self.author_blog1, 18, 20, 1, 2)
        self.grid.addWidget(self.author_github, 19, 20, 1, 2)

        self.load_model.clicked.connect(self.btnclicked)
        self.inf_img.clicked.connect(self.btnclicked)
        self.monitor_desktop.clicked.connect(self.btnclicked)
        self.monitor_off.clicked.connect(self.btnclicked)

        self.author_blog.clicked.connect(self.btnclicked)
        self.author_blog1.clicked.connect(self.btnclicked)
        self.author_github.clicked.connect(self.btnclicked)

        # self.inf_img.setShortcut(Qt.Key_Up)
        # self.monitor_desktop.setShortcut(Qt.Key_Down)

        self.inf_img.setEnabled(False)
        self.monitor_desktop.setEnabled(False)
        self.monitor_off.setEnabled(False)

    def btnclicked(self):
        '''
        处理按钮事件
        '''
        sender = self.sender()
        if sender == self.load_model:
            # open jpg/png dir
            dir_path = QFileDialog.getExistingDirectory(self, 'Open Dirs')
            if dir_path.strip() != '':
                self.do_load_model(dir_path)
        elif sender == self.inf_img:
            img_fp, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
            self.do_inf_img(img_fp)
        elif sender == self.monitor_desktop:
            self.do_monitor_desktop()
        elif sender == self.monitor_off:
            self.do_monitor_off()
        elif sender == self.author_blog:
            QDesktopServices.openUrl(QUrl(self.author_info['blog']))
        elif sender == self.author_blog1:
            QDesktopServices.openUrl(QUrl(self.author_info['blog1']))
        elif sender == self.author_github:
            QDesktopServices.openUrl(QUrl(self.author_info['github']))

    def do_inf_img(self, img_fp):
        if self.detector:
            # img_np = cv2.imread(img_fp)

            img_np = cv2.imdecode(np.fromfile(img_fp, dtype=np.uint8), -1)
            # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            bboxes, labels, _ = self.detector([img_np])[0]  # 模型预测
            for i in range(len(bboxes)):
                if bboxes[i][4] < self.visiual_thresh:
                    continue
                x1, y1, x2, y2, score = bboxes[i]
                cv2.rectangle(img_np, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(img_np, "{}:{:.3f}".format(coco_classes[labels[i]], score), (int(x1), int(y1)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            img_np_w, img_np_h = img_np.shape[:2]
            if img_np_h > self.window_size[1] or img_np_w > self.window_size[0]:
                img_np = cv2.resize(img_np, (self.window_size[0] - 10, self.window_size[1] - 10))

            img_np = QImage(img_np[:], img_np.shape[1], img_np.shape[0], img_np.shape[1] * 3,
                            QImage.Format_RGB888)
            self.window.setPixmap(QPixmap(img_np))

            # cv2.imshow('img', img_np)
            # cv2.waitKey(0)

    def do_monitor_desktop(self):
        self.monitor_tag = True
        while self.monitor_tag:
            primary_screen = QApplication.primaryScreen()  # 获取截图
            qimg = primary_screen.grabWindow(hwnd, 0, 0, self.desktop_width,
                                             self.desktop_height).toImage()  # 截图转换为QImage
            img_np = qimage2ndarray.rgb_view(qimg)  # QImage转换为np.ndarray
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            bboxes, labels, _ = self.detector([img_np])[0]  # 模型预测
            for i in range(len(bboxes)):
                if bboxes[i][4] < self.visiual_thresh:
                    continue
                x1, y1, x2, y2, score = bboxes[i]
                cv2.rectangle(img_np, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(img_np, "{}:{:.3f}".format(coco_classes[labels[i]], score), (int(x1) + 15, int(y1) + 15),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            img_np_w, img_np_h = img_np.shape[:2]
            if img_np_h > self.window_size[1] or img_np_w > self.window_size[0]:
                img_np = cv2.resize(img_np, (self.window_size[0] - 10, self.window_size[1] - 10))

            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_np = QImage(img_np[:], img_np.shape[1], img_np.shape[0], img_np.shape[1] * 3,
                            QImage.Format_RGB888)
            self.window.setPixmap(QPixmap(img_np))
            QApplication.processEvents()

    def do_monitor_off(self):
        self.monitor_tag = False
        self.window.setPixmap(self.bg)

    def do_load_model(self, dir_path):
        self.detector = None
        try:
            self.detector = Detector(dir_path, 'cuda', 0)
        except:
            pass
        if self.detector:
            self.status.setText(' 模型加载成功')
            self.status.setStyleSheet('background-color: rgb(120, 224, 143);')
            self.inf_img.setEnabled(True)
            self.monitor_desktop.setEnabled(True)
            self.monitor_off.setEnabled(True)
        else:
            self.status.setText(' 模型加载失败')
            self.status.setStyleSheet('background-color: rgb(229, 80, 57);')
            self.inf_img.setEnabled(False)
            self.monitor_desktop.setEnabled(False)
            self.monitor_off.setEnabled(False)
        self.update()

    def clicked(self):
        self.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Main()
    ex.show()
    app.exec_()
