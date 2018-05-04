#coding:utf-8
'''
日本程序员提供了源码，利用keras深度学习框架来训练自己的人脸识别模型。
keras是一个神经网络框架，纯python编写，被集成了tensorflow和theano这样的深度学习框架。
其存在的目的就是简化开发复杂度，能够让你迅速做出产品，更关键的是keras有中文文档。
这里使用keras的tensorflow版。
'''

'''
该程序可以指定要截取的人脸数量，由cv2.imwrite()函数完成实际的图片保存。
在获取图像的同时，在图像上提供了信息输出，显示当前已截取人脸图片的张数。
制作标签的过程：
我们准备2000张自己的不同角度可识别出来的人脸图片，单独存放在img目录下。
除此之外，至少还需要相同数量保存另一个人的人脸图像集，来区分本人与其他人。
保存要注意每一个目录下的所有图片一定是同一个人，否则可能造成识别判断错误的情况。
'''

import cv2
import sys
from PIL import Image

def catchImageFrame(window_name, camera_idx, path, img_max_storage):
    cv2.namedWindow(window_name)
    # 图像来源来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)
    # 使用OpenCV的人脸识别分类器
    classfier = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")

    # 识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)

    num = 0
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

                # 将当前帧保存为图片
                img_name = '%s/%d.jpg' % (path, num)
                image = frame[y - 10:y + h + 10, x - 10:x + w + 10]
                # imwrite:保存图片。如果目录不存在，则返回False。成功保存返回True
                cv2.imwrite(img_name, image)

                num += 1
                if num > img_max_storage:  # 如果超过指定的最大保存数量
                    break

                cv2.rectangle(frame, (x-10, y-10),
                              (x+w+10, y+h+10), color, 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                # putText:在图像上绘制文字
                cv2.putText(frame, 'num:%d' % (num), (x+30, y+30),
                            font, 1, (255, 0, 255), 4)
        else:
            print('不是人脸')

        if num > img_max_storage:  # 如果超过指定的最大保存数量
            break

        # 显示图像并等待按键输入，输入‘q’退出程序
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    catchImageFrame('识别人脸', 0, './data/test', 1000)

















