#cofing:utf-8

import os
import sys
import numpy as np
import cv2

IMAGE_SIZE = 64

# 按照指定图像大小调整尺寸
def resizeImg(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
	top, bottom, left, right = (0, 0, 0, 0)
	# 获取图像尺寸
	h, w, _ = image.shape
	# 对于长宽不相等的图片，找到最长的一边
	longest_edge = max(h, w)
	# 计算短边需要增加多上像素宽度使其与长边等长
	if h < longest_edge:
		dh = longest_edge - h
		top = dh // 2
		bottom = dh - top
	elif w < longest_edge:
		dw = longest_edge - w
		left = dw // 2
		right = dw - left
	else:
		print('pass')
		pass
	# RGB颜色
	BLACK = [0, 0, 0]
	# 给图像增加边界，使图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
	constant = cv2.copyMakeBorder(image, top, bottom, left, right,
								  cv2.BORDER_CONSTANT, value=BLACK)
	# 调整图像大小并返回
	return cv2.resize(constant, (height, width))

# 读取训练数据
images = []
labels = []
def readPath(path):
	for dir_item in os.listdir(path):
		# 从初始路径开始叠加，合并成可识别的操作路径
		full_path = os.path.abspath(os.path.join(path, dir_item))
		if os.path.isdir(full_path):	# 如果是文件夹，继续递归调用
			readPath(full_path)
		else:	# 文件
			print('dir_item: ', dir_item)
			if dir_item.endswith('.jpg'):
				print('full_path: ', full_path)
				image = cv2.imread(full_path)
				print('image:', image)
				image = resizeImg(image, IMAGE_SIZE, IMAGE_SIZE)

				# 放开这行代码，可以看到resizeImg()函数的实际调用效果
				# cv2.imwrite('1.jpg', image)
				images.append(image)
				labels.append(path)
			else:
				print('not jgp')
				continue
	return images, labels

# 从指定路径读取训练数据
def loadDataSet(path):
	images, labels = readPath(path)
	'''
	将输入的所有图片转化位4维数组，尺寸为
				(图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
	自己的图片一共1000张图片，IMAGE_SIZE为64(可设置不同的值)，所以尺寸应为：
				(1000*64*64*3)
	图片为64*64像素，一个像素3个颜色值(RGB)
	'''
	images = np.array(images)
	# 标注数据，'me'文件夹下都是我自己的人脸图片，全部指定为0，另一个文件夹下都是另一个人的，全部指定为1
	labels = np.array([0 if label.endswith('me') else 1 for label in labels])

	return images, labels

if __name__ == '__main__':
	sys.setrecursionlimit(1000000)
	images, labels = loadDataSet('./data')
	print(images.shape)
	print(labels)
	print(labels.shape)