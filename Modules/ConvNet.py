import tensorflow as tf
from six.moves import xrange


def __conv1d__(inputs, width, stride, in_channels, out_channels):
    filter_1d = tf.get_variable(name='filter', shape=(width, in_channels, out_channels))
    return tf.nn.conv1d(inputs, filter_1d, stride, 'SAME')

def __conv1d_alone_time__(inputs, width, in_channels, out_channels):
    return __conv1d__(inputs, width, 1, in_channels, out_channels)

class Conv1dBankWithMaxPool(object):
    """Conv1d Bank.
    The output is max_pooled along time.
    """

    def __init__(self, K, activation=tf.nn.relu):
        self.__K = K
        self.__activation = activation

    @property
    def K(self):
        return self.__K

    @property
    def activation(self):
        return self.__activation

    def __call__(self, inputs, is_training=True, scope=None):
        """
        Args:
            inputs: with shape -> (batch_size, time_step/width, units/channels)
        """
        with tf.variable_scope(scope or type(self).__name__):
            in_channels = inputs.shape[-1].value
            conv_lst = []
            for idk in xrange(1, self.K + 1):
                with tf.variable_scope('inner_conv_%d' % idk):
                    conv_k = self.activation(__conv1d_alone_time__(inputs, idk, in_channels, in_channels))
                    norm_k = tf.contrib.layers.batch_norm(conv_k, center=True, scale=True, is_training=is_training)
                conv_lst.append(norm_k)

            stacked_conv = tf.stack(conv_lst, axis=-1)   # shape -> (batch_size, time_step/width, units/channels, K/height)
            #re_shape = tf.shape(stacked_conv)[:2] + [1, in_channels * self.K]
            re_shape = [tf.shape(stacked_conv)[0], tf.shape(stacked_conv)[1], 1, in_channels * self.K]
            stacked_conv = tf.reshape(stacked_conv, shape=re_shape)     # shape -> (batch_size, time_step/width, 1, units*K/channels)

            ### max pool along time
            ksize = [1, 2, 1, 1]
            strid = [1, 1, 1, 1]
            pooled_conv = tf.squeeze(tf.nn.max_pool(stacked_conv, ksize, strid, 'SAME'), axis=2)    # shape -> (batch_size, time_step/width, units*K/channels)

            return pooled_conv

class Conv1dProjection(object):
    """Conv1d Projection
    """

    def __init__(self, proj_unit, width=3, activation=tf.nn.relu):
        self.__proj_unit = proj_unit
        self.__width = width
        self.__activation = activation

    @property
    def proj_unit(self):
        return self.__proj_unit

    @property
    def width(self):
        return self.__width

    @property
    def activation(self):
        return self.__activation

    def __call__(self, inputs, is_training=True, scope=None):
        """
        Args:
            inputs: with shape -> (batch_size, time_step/width, units/channels)
        """
        with tf.variable_scope(scope or type(self).__name__):
            filter_width = self.width
            proj_0 = self.proj_unit[0]
            proj_1 = self.proj_unit[1]
            in_channels = inputs.get_shape()[-1].value
            with tf.variable_scope('inner_conv_with_acti'):
                conv_a = self.activation(__conv1d_alone_time__(inputs, filter_width, in_channels, proj_0))
                norm_a = tf.contrib.layers.batch_norm(conv_a, center=True, scale=True, is_training=is_training)
            with tf.variable_scope('inner_conv_linear'):
                conv_l = __conv1d_alone_time__(norm_a, filter_width, proj_0, proj_1)
                norm_l = tf.contrib.layers.batch_norm(conv_l, center=True, scale=True, is_training=is_training)

            return norm_l

