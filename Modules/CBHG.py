import tensorflow as tf
from TFAttention.RNNCell import GRUCell
from TFAttention.DynamicRNNScan import biDynamicRNNScan
from Tacotron.Modules import ConvNet, HighwayNet

Conv1dBankWithMaxPool   = ConvNet.Conv1dBankWithMaxPool
Conv1dProjection        = ConvNet.Conv1dProjection
FCHighwayNet            = HighwayNet.FCHighwayNet

class CBHG(object):
    """CBHG Net
    """
    
    def __init__(self, bank_K, proj_unit):
        """
        Args:
            bank_K: int
            proj_unit: a pair of int
        """
        self.__bank_K = bank_K
        self.__proj_unit = proj_unit

    @property
    def bank_K(self):
        return self.__bank_K

    @property
    def proj_unit(self):
        return self.__proj_unit

    def __call__(self, inputs, time_major=None):
        assert time_major is not None, "[*] You must specify whether is time_major or not!"
        if time_major:
            inputs = tf.transpose(inputs, perm=(1,0,2))
        assert inputs.get_shape()[-1] == self.proj_unit[1], "[!] input's shape is not the same as ConvProj's output!"
        ConvBankWithPool    = Conv1dBankWithMaxPool(self.bank_K)
        ConvProj            = Conv1dProjection(self.proj_unit)
        Highway             = FCHighwayNet(4)
        rnn_cell_fw         = GRUCell(self.proj_unit[1])
        rnn_cell_bw         = GRUCell(self.proj_unit[1])

        ### calculate
        # conv net
        output_0 = ConvBankWithPool(inputs)
        output_1 = ConvProj(output_0)
        res_output = tf.identity(inputs) + output_1

        # highway net
        trans_out = tf.transpose(res_output, perm=(1,0,2))
        max_time_steps = tf.shape(trans_out)[0]
        batch_size = tf.shape(trans_out)[1]
        time = tf.constant(0, dtype=tf.int32)
        cond = lambda time, *_: tf.less(time, max_time_steps)
        output_ta = tf.TensorArray(size=max_time_steps, dtype=tf.float32)
        def body(time, output_ta):
            output = Highway(trans_out[time])
            output_ta = output_ta.write(time, output)
            return tf.add(time, 1), output_ta
        _, final_output_ta = tf.while_loop(cond, body, [time, output_ta])
        highway_output = tf.reshape(final_output_ta.stack(), shape=(max_time_steps, batch_size, self.proj_unit[1]))

        # biGRU
        final_output, *_ = biDynamicRNNScan(rnn_cell_fw, rnn_cell_bw, highway_output)
        if not time_major:
            final_output = tf.transpose(final_output, perm=(1,0,2))

        return final_output

