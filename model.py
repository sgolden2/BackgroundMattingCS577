from keras import Model
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.applications import ResNet101
from keras.utils import conv_utils
import tensorflow as tf

IN_CHANNELS = 6  # RGB for self.background and image
	b1=DepthwiseConv2D((3,3),dilation_rate=(6,6),padding="same",use_bias=False)(x)
	b1=BatchNormalization()(b1)
OUT_STRIDE = 3
BATCH_SIZE = 10

MOMENTUM = 0.1
EPSILON = 1e-5


class Backbone(Layer):
	def __init__(self, name="backbone", **kwargs)
		super(Backbone, self).__init__(name=name, **kwargs)
		self.resnet = ResNet101(
			include_top=False,
			weights='imagenet',
			input_tensor=None,
			input_shape=input_shape,
			pooling=None,
			classes=2,
		)

	def call(self, inputs, training=None):
		return self.resnet(inputs)


class ASPP(Model):
	def __init__(self, filters, dilations=[3,6,9]):
		super().__init__()

		# convolutions
		self.conv1 = Conv2D(filters, 1, padding='SAME', dilation_rate=1, use_bias=False)
		self.bn1 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)
		self.relu1 = ReLU()
		self.conv2 = Conv2D(filters, 3, padding='SAME', dilation_rate=dilations[0], use_bias=False)
		self.bn2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)
		self.relu2 = ReLU()
		self.conv3 = Conv2D(filters, 3, padding='SAME', dilation_rate=dilations[1], use_bias=False)
		self.bn3 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)
		self.relu3 = ReLU()
		self.conv4= Conv2D(filters, 3, padding='SAME', dilation_rate=dilations[2], use_bias=False)
		self.bn4= BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)
		self.relu4= ReLU()

		# pooling
		self.pooling = GlobalAveragePooling2D()
		self.conv5 = Conv2D(filters, 1, use_bias=False)
		self.bn5 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)
		self.relu5 = ReLU()
		
		# aspp output
		self.output = Sequential([
			Conv2D(filters, 1, use_bias=False),
			BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON),
			ReLU(),
			Dropout(0.1)
		])

	def call(self, inputs, training=None):
		convblock1 = self.conv1(inputs, training=training)
		convblock1 = self.bn1(convblock1, training=training)
		convblock1 = self.relu1(convblock1, training=training)

		convblock2 = self.conv2(inputs, training=training)
		convblock2 = self.bn2(convblock2, training=training)
		convblock2 = self.relu2(convblock2, training=training)

		convblock3 = self.conv3(inputs, training=training)
		convblock3 = self.bn3(convblock3, training=training)
		convblock3 = self.relu3(convblock3, training=training)

		convblock4 = self.conv4(inputs, training=training)
		convblock4 = self.bn4(convblock4, training=training)
		convblock4 = self.relu4(convblock4, training=training)

		poolblock = self.poolling(inputs, training=training)
		poolblock = poolblock[:,None,None,:]
		poolblock = self.conv5(poolblock, training=training)
		poolblock = self.bn5(poolblock, training=training)
		poolblock = self.relu5(poolblock, training=training)
		poolblock = tf.image.resize(poolblock, (tf.shape(inputs)[1], tf.shape(inputs)[2]), 'nearest')

		pyramid = tf.concat([convblock1, convblock2, convblock3, convblock4, poolblock], axis=-1)

		return self.output(pyramid, training=training)



class Decoder(Model):
	def __init__(self, channels):
		super().__init__()

		self.convs = [Conv2D(channels[i], 3, padding='SAME', use_bias=False) for i in range(0,4)]
		self.bns = [BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON) for _ in range(len(self.convs) - 1)]
		self.relu = ReLU()

	def call(self, inputs, training=None):
		x4,x3,x2,x1,x0 = inputs
		# refer to https://github.com/PeterL1n/BackgroundMattingV2-TensorFlow/blob/master/model/decoder.py
		x = tf.image.resize(x4, tf.shape(x3)[1:3])




class MattingModel(Model):
	def __init__(self, input_shape):
		self.backbone = Backbone()
		self.aspp = ASPP()
		self.decoder = Decoder()

	def call(self, inputs):
		x = self.backbone(inputs)
		x = self.aspp(x)
		x = self.decoder(x)
		return x


### MAIN ###
m = MattingModel((1920,1080,3))
