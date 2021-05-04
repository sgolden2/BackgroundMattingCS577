from keras import Model
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.applications import ResNet101
from keras.utils import conv_utils
import tensorflow as tf
from resnet import ResNetEncoder


IN_CHANNELS = 6  # RGB for self.background and image
OUT_STRIDE = 3
BATCH_SIZE = 10

MOMENTUM = 0.1
EPSILON = 1e-5


class ASPP(Layer):
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
        self.combinator = Sequential([
            Conv2D(filters, 1, use_bias=False),
            BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON),
            ReLU(),
            Dropout(0.1)
        ])

    def call(self, inputs, training=True):
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

        poolblock = self.pooling(inputs, training=training)
        poolblock = poolblock[:,None,None,:]
        poolblock = self.conv5(poolblock, training=training)
        poolblock = self.bn5(poolblock, training=training)
        poolblock = self.relu5(poolblock, training=training)
        poolblock = tf.image.resize(poolblock, (tf.shape(inputs)[1], tf.shape(inputs)[2]), 'nearest')

        pyramid = tf.concat([convblock1, convblock2, convblock3, convblock4, poolblock], axis=-1)

        return self.combinator(pyramid, training=training)


class Decoder(Layer):
    def __init__(self, channels):
        super().__init__()

        self.convs = [Conv2D(channels[i], 3, padding='SAME', use_bias=False) for i in range(0,4)]
        self.bns = [BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON) for _ in range(len(self.convs) - 1)]
        self.relu = ReLU()

    def call(self, inputs, training=True):
        pyr4,pyr3,pyr2,pyr1,pyr0 = inputs

        x = tf.image.resize(pyr4, tf.shape(pyr3)[1:3])
        x = tf.concat([x, pyr3], axis=-1)
        x = self.convs[0](x, training=training)
        x = self.bns[0](x, training=training)
        x = self.relu(x, training=training)

        x = tf.image.resize(x, tf.shape(pyr2)[1:3])
        x = tf.concat([x, pyr2], axis=-1)
        x = self.convs[1](x, training=training)
        x = self.bns[1](x, training=training)
        x = self.relu(x, training=training)

        x = tf.image.resize(x, tf.shape(pyr1)[1:3])
        x = tf.concat([x, pyr1], axis=-1)
        x = self.convs[2](x, training=training)
        x = self.bns[2](x, training=training)
        x = self.relu(x, training=training)

        x = tf.image.resize(x, tf.shape(pyr0)[1:3])
        x = tf.concat([x, pyr0], axis=-1)
        x = self.convs[3](x, training=training)

        return x


class MattingModel(Model):
    def __init__(self):
        super(MattingModel, self).__init__()
        self.backbone = ResNetEncoder("resnet101")
        self.aspp = ASPP(256, [3, 6, 9])
        self.decoder = Decoder([128, 64, 48, (1 + 3 + 1 + 32)])

    def call(self, inputs, training=True, _output_foreground_as_residual=False):
        source, background = inputs
        x = tf.concat([source, background], axis=-1)

        x, *skips = self.backbone(x, training=training)
        x = self.aspp(x, training=training)
        x = self.decoder([x, *skips], training=training)

        alpha = tf.clip_by_value(x[:, :, :, 0:1], 0, 1)
        foreground = x[:, :, :, 1:4]
        error_map = tf.clip_by_value(x[:, :, :, 4:5], 0, 1)
        hidden_features = tf.nn.relu(x[:, :, :, 5:])

        if not _output_foreground_as_residual:
            foreground = tf.clip_by_value(foreground + source, 0, 1)

        return alpha, foreground, error_map, hidden_features


