# FaceRecognition_keras
基于keras，CNN网络的人脸识别程序
## 目录结构说明：
```
|————————————————————
|bin:   存放最终执行的文件
|  |_____keras_face_recongnition.py: 采集摄像头图片帧，并使用CNN模预测，识别人脸是不是本人
|  |_____train_model.py：CNN网络模型训练
|
|data:  存放采集到的图片，me目录下是自己，other目录下是另一个人
|preprocessing-module.py：    预处理模块，单独执行，主要用于采集图像，并保存在data下的指定目录中
|src：   程序关键模块的源文件
|  |_____DataSetHandling.py：主要用于加载图片文件，划分数据集，one-hot编码和图像数据归一化等操作
|  |_____ImageAcq.py：从摄像头实时采集图片帧
|  |_____LoadFaceDataset.py：从指定路径读取已经预处理好的图片，并调整图片大小
|  |_____ModelFit.py：搭建CNN网络，模型训练、保存和加载操作
|  |_____utils.py：按字段读取配置文件内容
|test:   功能性测试
|config.ini：配置文件，对摄像头设备编号、人脸数据路径、CNN模型路径做配置
|haarcascade_frontalface_alt2.xml:   opencv中的人脸数据集之一
|me_face.model.h5:   keras训练的本人人脸识别模型
|————————————————————
```
