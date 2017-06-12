import tensorflow as tf
from six.moves import xrange

class FCHighwayNet(object):
    """Implements Highway Networks.
  
    Rupesh Kumar Srivastava, Klaus Greff, Ju ̈rgen Schmidhuber.
    "Highway Networks."
    https://arxiv.org/abs/1505.00387
    """

    def __init__(self, layer_num, activation=tf.nn.relu):
        self.__layer_num = layer_num
        self.__activation = activation

    @property
    def layer_num(self):
        return self.__layer_num

    @property
    def activation(self):
        return self.__activation

    def __flow_layer(self, inputs, units):
        H = tf.layers.dense(name='H', inputs=inputs, units=units, activation=self.activation)
        T = tf.layers.dense(name='T', inputs=inputs, units=units, activation=tf.sigmoid)
        y  = H * T  +  inputs * (1 - T)
        return y

    def __call__(self, inputs, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            x_l = inputs
            units = inputs.shape[-1].value
            for idx in xrange(1, self.layer_num + 1):
                with tf.variable_scope('inner_layer_%d' % idx):
                    y_l = self.__flow_layer(x_l, units)
                    x_l = y_l
            
            return y_l

