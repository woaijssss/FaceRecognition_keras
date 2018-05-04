#coding:utf-8

import cv2

import src.utils as utils
from src.ModelFit import CNNModel
from src.ImageAcq import FaceCollection

if __name__ == '__main__':
	config_path = '../config.ini'
	camera_id = utils.readConfig(config_path, 'model', 'camera_id')
	mode_path = utils.readConfig(config_path, 'model', 'mode_path')
	# 人脸识别分类器本地存储路径
	data_path = utils.readConfig(config_path, 'model', 'data_path')
	# 加载模型
	model = CNNModel()
	model.loadModel(path=mode_path)

	# 框住人脸的矩形边框颜色
	color = (0, 255, 0)

	fc = FaceCollection(face_data_path=data_path, camera_id=camera_id)
	# 循环检测识别人脸
	while True:
		face_rects, frame = fc.faceCollection()
		if len(face_rects) > 0:
			for face_rect in face_rects:
				x, y, w, h = face_rect
				# 截取脸部图像提交给模型识别这是谁
				image = frame[y-10 : y+h+10, x-10 : x+w+10]
				face_id = model.facePredict(image)

				# 如果是“我”，用方框框出，并用文字提示
				print('face_id:', face_id)
				if not face_id:
					cv2.rectangle(frame, (x-10, y-10),
								  (x+w+10, y+h+10), color, thickness=2)
					cv2.putText(frame, 'Me', (x+30, y+30),
								cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
				else:
					pass
		cv2.imshow('识别自己', frame)
		c = cv2.waitKey(10)
		if c & 0xFF == ord('q'):
			break
	del fc
	cv2.destroyAllWindows()