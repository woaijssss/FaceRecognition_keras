#-*- coding: utf-8 -*-

'''
从实时视频流中识别出人脸区域，从原理上看，依然属于机器学习的领域之一。本质上与google
利用深度学习识别出猫没有区别。
程序通过大量的人脸图片数据进行训练，利用数学算法建立可靠的人脸特征模型，即可识别出人脸。
这些工作在opencv中也可以做，我们只需要调用对应的API接口即可。
'''

import cv2
import sys
from PIL import Image   # 图像处理包Pillow

def catchImageFrame(window_name, camera_idx):
    cv2.namedWindow(window_name)
    # 图像来源来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)
    # 使用OpenCV的人脸识别分类器
    classfier = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")

    # 识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)

    while cap.isOpened():
        ret, frame = cap.read()  # 读取一帧数据
        if not ret:
            break

        # gray和frame都是ndarray类型
        # 将当前帧转换成灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # 灰度转换

        # 人脸检测，1.2和2分别为图片的缩放比例和需要检测的有效点数
        '''
        detectMultiScale:http://blog.csdn.net/yanrong1095/article/details/78685390
        '''
        face_rects = classfier.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32)
        )
        if (len(face_rects)):   # 检测到人脸(可检测多个)
            for face_rect in face_rects:    # 单独框出每一个人脸
                x, y, w, h = face_rect
                cv2.rectangle(frame, (x-10, y-10),
                              (x+w+10, y+h+10), color, 2)
        else:
            print('不是人脸')

        # 显示图像并等待按键输入，输入‘q’退出程序
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    catchImageFrame('识别人脸',0)
