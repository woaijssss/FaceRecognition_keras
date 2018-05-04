# -*- coding: utf-8 -*-

import cv2
import sys
from PIL import Image


def catchImageFrame(window_name, camera_idx):
    cv2.namedWindow(window_name)

    # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)

    while cap.isOpened():
        ret, frame = cap.read()  # 读取一帧数据
        if not ret:
            break

        # gray和frame都是ndarray类型
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 灰度转换
            # 显示图像并等待10毫秒按键输入，输入‘q’退出程序
        cv2.imshow(window_name, gray)
        c = cv2.waitKey(0)
        if c & 0xFF == ord('q'):
            break

            # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    catchImageFrame("capture_frame", 0)