import tensorflow as tf
from keras import Model
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.applications import ResNet101
import keras.backend as K


#tf.config.experimental_run_functions_eagerly(True)
tf.compat.v1.disable_eager_execution() 

MOMENTUM = 0.15
EPSILON = 1e-5
DROPOUT_RATE = 0.15

ASPP_DILATIONS = [3,6,9]
ASPP_FILTERS = 256

DECODER_CHANNELS = [128, 64, 48, 37]

BASE_MODEL_SCALE = float(1/4)

REFINER_CHANNELS = [24, 16, 12, 4]


def base_alpha_mse_loss(pha_true, preds):
	pha_pred = tf.clip_by_value(preds[:, :, :, 0:1], 0, 1)
	pha_true = tf.cast(tf.reshape(pha_true, tf.shape(pha_pred)), tf.float32)
	return tf.math.squared_difference(pha_true, pha_pred)


def full_alpha_mse_loss(pha_true, pha_pred):
	pha_true = tf.cast(tf.reshape(pha_true, tf.shape(pha_pred)), tf.float32)
	return tf.math.squared_difference(pha_true, pha_pred)


def get_base_model(input_shape=(768,432,6), compiled=True):

	### ResNet Backbone ###

	inp = Input(shape=input_shape)
	img,bgr = Lambda(lambda x: tf.split(x,2,axis=-1))(inp)
	#reduced_channels = Conv3D(filters=512, kernel_size=(3,3,2),padding='SAME')(inp)

	resnet = ResNet101(
		weights='imagenet', 
		include_top=False,
		input_tensor=img
	)

	resnet.trainable=False
	backbone_in = resnet.input
	resblock1 = resnet.get_layer('conv1_relu').output
	resblock2 = resnet.get_layer('conv2_block3_out').output
	resblock3 = resnet.get_layer('conv3_block4_out').output
	backbone_out = resnet.output

	x = backbone_out
	#x = Add()([img_res,bgr_res])

	### ASPP ###

	# conv block 1
	conv1 = Conv2D(ASPP_FILTERS, 1, padding='SAME', dilation_rate=1, use_bias=False, name='aspp_conv1_in')(x)
	conv1 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(conv1)
	conv1 = ReLU()(conv1)

	# conv block 2
	conv2 = Conv2D(ASPP_FILTERS, 3, padding='SAME', dilation_rate=ASPP_DILATIONS[0], use_bias=False, name='aspp_conv2_in')(x)
	conv2 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(conv2)
	conv2 = ReLU()(conv2)

	# conv block 3
	conv3 = Conv2D(ASPP_FILTERS, 3, padding='SAME', dilation_rate=ASPP_DILATIONS[1], use_bias=False, name='aspp_conv3_in')(x)
	conv3 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(conv3)
	conv3 = ReLU()(conv3)

	# conv block 4
	conv4 = Conv2D(ASPP_FILTERS, 3, padding='SAME', dilation_rate=ASPP_DILATIONS[2], use_bias=False, name='aspp_conv4_in')(x)
	conv4 = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(conv4)
	conv4 = ReLU()(conv4)

	# pooling block
	dims = tf.shape(backbone_out)[1],tf.shape(x)[2]
	pool = GlobalAveragePooling2D(name='aspp_pool_in')(x)
	pool = pool[:,None,None,:]
	pool = Conv2D(ASPP_FILTERS, 1, use_bias=False)(pool)
	pool = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(pool)
	pool = ReLU()(pool)
	pool = tf.image.resize(pool, dims, 'nearest')

	# pyramid construction
	pyr = tf.concat([conv1,conv2,conv3,conv4,pool], axis=-1)
	pyr = Conv2D(ASPP_FILTERS, 1, use_bias=False)(pyr)
	pyr = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(pyr)
	pyr = ReLU()(pyr)
	pyr = Dropout(DROPOUT_RATE)(pyr)

	### DECODER ###

	x4,x3,x2,x1,x0 = pyr, resblock3, resblock2, resblock1, backbone_in

	x = Lambda(lambda a: tf.image.resize(a, tf.shape(x3)[1:3]))(x4)
	x = tf.concat([x,x3],axis=-1)
	x = Conv2D(DECODER_CHANNELS[0], 3, padding='SAME', use_bias=False)(x)
	x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x)
	x = ReLU()(x)

	x = Lambda(lambda a: tf.image.resize(a, tf.shape(x2)[1:3]))(x)
	x = tf.concat([x,x2],axis=-1)
	x = Conv2D(DECODER_CHANNELS[1], 3, padding='SAME', use_bias=False)(x)
	x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x)
	x = ReLU()(x)

	x = Lambda(lambda a: tf.image.resize(a, tf.shape(x1)[1:3]))(x)
	x = tf.concat([x,x1],axis=-1)
	x = Conv2D(DECODER_CHANNELS[2], 3, padding='SAME', use_bias=False)(x)
	x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x)
	x = ReLU()(x)

	x = Lambda(lambda a: tf.image.resize(a, tf.shape(x0)[1:3]))(x)
	x = tf.concat([x,x0],axis=-1)
	x = Conv2D(DECODER_CHANNELS[3], 3, padding='SAME', use_bias=True)(x)

	print(x.shape)

	out = x

	### COMPILE ###

	model = Model(inputs=inp, outputs=out)

	if compiled:
		model.compile(optimizer='adam', loss=base_alpha_mse_loss)

	return model


def get_full_model(input_shape=(768,432,6), compiled=True):
	full_dims = (int(input_shape[0]), int(input_shape[1]))
	half_dims = (int(input_shape[0] * (1/2)), int(input_shape[1] * (1/2)))
	quart_dims = (int(input_shape[0] * (1/4)), int(input_shape[1] * (1/4)))
	coarse_dims = (int(input_shape[0] * BASE_MODEL_SCALE), int(input_shape[1] * BASE_MODEL_SCALE))

	inp = Input(shape=input_shape)
	img,bgr = Lambda(lambda x: tf.split(x,2,axis=-1))(inp)
	
	coarse_img = tf.image.resize(img, coarse_dims)
	coarse_bgr = tf.image.resize(bgr, coarse_dims)

	base_model_out = get_base_model(input_shape=coarse_dims+(6,), compiled=False)(tf.concat([coarse_img,coarse_bgr],axis=-1))

	alpha, forgr = tf.clip_by_value(base_model_out[:, :, :, 0:1], 0, 1), base_model_out[:, :, :, 1:4]
	error, hiddn = tf.clip_by_value(base_model_out[:, :, :, 4:5], 0, 1), tf.nn.relu(base_model_out[:, :, :, 5:])

	trim_err = tf.concat([hiddn, alpha, forgr], axis=-1)
	orig = tf.concat([img,bgr], axis=-1)

	sample_feats = tf.image.resize(trim_err, half_dims)
	sample_image = tf.image.resize(orig, half_dims)
	sample = tf.concat([sample_feats,sample_image], axis=-1)
	sample = tf.pad(sample, 
		tf.constant([
			[0, 0], 
			[3, 3], 
			[3, 3], 
			[0, 0]]
		)
	)

	x = Conv2D(REFINER_CHANNELS[0], 3, use_bias=False)(sample)
	x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x)
	x = ReLU()(x)
	x = Conv2D(REFINER_CHANNELS[1], 3, use_bias=False)(sample)
	x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x)
	x = ReLU()(x)
	x = tf.image.resize(x, (full_dims[0]+4,full_dims[1]+4), 'nearest')
	sample = tf.pad(orig, 
		tf.constant([
			[0, 0], 
			[2, 2], 
			[2, 2], 
			[0, 0]]
		)
	)

	x = tf.concat([x,sample], axis=-1)
	x = Conv2D(REFINER_CHANNELS[2], 3, use_bias=False)(x)
	x = BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)(x)
	x = ReLU()(x)
	x = Conv2D(REFINER_CHANNELS[3], 3, use_bias=True)(x)

	alpha = tf.clip_by_value(x[:,:,:,0], 0, 1), 
	#foreground = tf.clip_by_value(x[:,:,:,1:]+img, 0, 1)
	#refined = tf.ones((full_dims[0], 1, quart_dims[1], quart_dims[0]), dtype=img.dtype)

	#out = tf.concat([alpha,foreground], axis=-1)
	out = alpha

	model = Model(inputs=inp, outputs=out)

	if compiled:
		model.compile(optimizer='adam', loss=full_alpha_mse_loss)

	return model
