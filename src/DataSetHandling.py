
import random
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import np_utils

from src.LoadFaceDataset import loadDataSet, IMAGE_SIZE

class DataSet:
	def __init__(self, path):
		# 训练集
		self.train_images = None
		self.train_labels = None

		# 验证集
		self.valid_images = None
		self.valid_labels = None

		# 测试集
		self.test_images = None
		self.test_labels = None

		# 数据集加载路径
		self.path = path

		# 当前库采用的维度顺序
		self.input_shape = None

	# 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理操作
	'''预处理这里做了4项工作：
	（1）按照交叉验证(train_test_split)的原则将数据集划分为：训练集、测试集和验证集
	（2）按照keras库运行的后端系统要求改变图像数据的维度顺序
	（3）将数据标签进行one-hot编码，使其向量化
	（4）归一化图像数据
	'''
	def load(self, img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE,
			 img_channels = 3, nb_classes = 2):
		# 加载数据集到内存
		images, labels = loadDataSet(self.path)

		train_images, valid_images, train_labels, valid_labels = train_test_split(
			images, labels, test_size=0.3, random_state=random.randint(0, 100)
		)
		_, test_images, _, test_labels = train_test_split(
				images, labels, test_size=0.5, random_state=random.randint(0, 100)
		)

		# 当前的维度顺序如果为'th'，则输入图片数据时的顺序为:channels,rows,cols；否则为:rows,cols,channels
		# 这部分代码是根据keras库要i求的维度顺序重组训练数据集
		# keras建立在tensorflow或theano基础上
		# image_dim_ordering():确定后端系统的类型
		if K.image_dim_ordering() == 'th':	# ‘th’代表theano，'tf'代表tensorflow
			train_images = train_images.reshape(
				train_images.shape[0], img_channels, img_rows, img_cols
			)
			valid_images = valid_images.reshape(
				valid_images.shape[0], img_channels, img_rows, img_cols
			)
			test_images = test_images.reshape(
				test_images.shape[0], img_channels, img_rows, img_cols
			)
			self.input_shape = (img_channels, img_rows, img_cols)
		else:
			train_images = train_images.reshape(
				train_images.shape[0], img_rows, img_cols, img_channels
			)
			valid_images = valid_images.reshape(
				valid_images.shape[0], img_rows, img_cols, img_channels
			)
			test_images = test_images.reshape(
				test_images.shape[0], img_rows, img_cols, img_channels
			)
			self.input_shape = (img_rows, img_cols, img_channels)

		# 输出训练集、验证集、测试集的数量
		print(train_images.shape[0], 'train samples')
		print(valid_images.shape[0], 'valid samples')
		print(test_images.shape[0], 'test samples')

		# 我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
		'''
		categorical_crossentropy():称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列
		categorical_crossentropy()函数要求标签集必须采用one-hot编码形式
		'''
		# 类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
		train_labels = np_utils.to_categorical(train_labels, nb_classes)
		valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
		test_labels = np_utils.to_categorical(test_labels, nb_classes)

		'''
		数据集先浮点后归一化的目的是提升网络收敛速度，减少训练时间，
		同时适应值域在（0,1）之间的激活函数，增大区分度。
		其实归一化有一个特别重要的原因是确保特征值权重一致。
		'''
		# 像素数据浮点化以便归一化
		train_images = train_images.astype('float32')
		valid_images = valid_images.astype('float32')
		test_images = test_images.astype('float32')

		# 将其归一化,图像的各像素值归一化到0~1区间
		train_images /= 255
		valid_images /= 255
		test_images /= 255

		self.train_images = train_images
		self.valid_images = valid_images
		self.test_images = test_images
		self.train_labels = train_labels
		self.valid_labels = valid_labels
		self.test_labels = test_labels
