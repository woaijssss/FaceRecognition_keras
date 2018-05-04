
import cv2

class FaceCollection:
	def __init__(self,
				 face_data_path='../haarcascade_frontalface_alt2.xml',
				 camera_id = 0
				 ):
		self.face_data_path = face_data_path
		self.camera_id = 0
		self.cap = cv2.VideoCapture(self.camera_id)

	def __del__(self):
		print('del capture')
		# 释放摄像头并销毁所有窗口
		self.cap.release()

	def faceCollection(self):
		# 捕获指定摄像头的实时视频流
		ret, frame = self.cap.read()  # 读取一帧视频
		if not ret:
			raise ret
		# 图像转灰度图，降低计算复杂度
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		# 使用人脸识别分类器，读入分类器
		cascade_classifer = cv2.CascadeClassifier(self.face_data_path)
		# 利用分类器识别出哪个区域为人脸
		face_rects = cascade_classifer.detectMultiScale(
			frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32)
		)
		return face_rects, frame
