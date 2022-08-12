import os
import sys
import time
import cv2
import qimage2ndarray
from mmdeploy_python import Detector
import win32gui
from PyQt5.QtWidgets import QApplication

# print(dir(Detector))
# print(help(Detector))

# 超参数 parameters
window_name = 'screen'
window_x1 = 0
window_y1 = 300
window_height = 780
window_width = 1400
root_dir = os.getcwd()
iou_thresh = 0.3
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']

# Model init
model_path = os.path.join(root_dir, 'yolox_s')
detector = Detector(model_path, 'cuda', 0)

hwnd = win32gui.FindWindow(None, 'C:\Windows\system32\cmd.exe')
app = QApplication(sys.argv)


def inference(img_np):
    bboxes, labels, _ = detector([img_np])[0]
    return bboxes, labels


def main():
    fps = 0
    tag = 0

    while True:
        t = time.time()

        primary_screen = QApplication.primaryScreen()  # 获取截图
        # 4通道转3通道
        # img_np = qimage2ndarray.rgb_view(primary_screen.grabWindow(hwnd).toImage())

        Qimg = primary_screen.grabWindow(hwnd, window_x1, window_y1, window_width,
                                         window_height).toImage()  # 截图转换为QImage
        t1 = time.time()
        img_np = qimage2ndarray.rgb_view(Qimg)  # QImage转换为np.ndarray
        t2 = time.time()
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        t3 = time.time()

        # inference
        dets, labels = inference(img_np)  # 模型预测
        t4 = time.time()

        print("QImage2ndarray:", t1 - t, "cv2.cvtColor:", t2 - t1, "inference:", t4 - t3)

        # img_res = results.imgs[0]
        cost_time = time.time() - t  # 计算耗时
        tag += 1
        if tag % 5 == 0:
            fps = format(1 / cost_time, '.2f')

        for i, j in zip(dets, labels):
            x1, y1, x2, y2, score = i
            if float(score) < iou_thresh:
                continue
            x1, y1, x2, y2, score = int(x1), int(y1), int(x2), int(y2), str(format(float(score), '.3f'))
            cls = classes[j]
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_np, f"{cls}:{score}", (x1 + 5, y1 + 18), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 0), 2)

        cv2.rectangle(img_np, (window_width - 150, 5), (window_width - 5, 40), (255, 255, 255), -1)
        cv2.putText(img_np, f'FPS:{fps}', (window_width - 145, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (82, 82, 255), 2)
        cv2.imshow(window_name, img_np)

        if (cv2.waitKey(1) & 0xFF) == ord('1'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
