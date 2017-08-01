import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn import FusedRNNCellAdaptor, TimeReversedFusedRNN
from TFCommon.RNNCell import FastGRUCell as GRUCell
from Tacotron.Modules import ConvNet, HighwayNet

bidirectional_dynamic_rnn = tf.nn.bidirectional_dynamic_rnn

Conv1dBankWithMaxPool   = ConvNet.Conv1dBankWithMaxPool
Conv1dProjection        = ConvNet.Conv1dProjection
FCHighwayNet            = HighwayNet.FCHighwayNet

class CBHG(object):
    """CBHG Net
    """
    
    def __init__(self, bank_K, proj_unit, highway_layers=4):
        """
        Args:
            bank_K: int
            proj_unit: a pair of int
        """
        self.__bank_K = bank_K
        self.__proj_unit = proj_unit
        self.__highway_layers = highway_layers

    @property
    def bank_K(self):
        return self.__bank_K

    @property
    def proj_unit(self):
        return self.__proj_unit

    @property
    def highway_layers(self):
        return self.__highway_layers

    def __call__(self, inputs, sequence_length=None, is_training=True, time_major=None):
        assert time_major is not None, "[*] You must specify whether is time_major or not!"
        if time_major:
            inputs = tf.transpose(inputs, perm=(1, 0, 2))     # Use batch major data.
        assert inputs.get_shape()[-1] == self.proj_unit[1], "[!] input's shape is not the same as ConvProj's output!"

        ### for correctness.
        if sequence_length is not None:
            mask = tf.expand_dims(array_ops.sequence_mask(sequence_length, tf.shape(inputs)[1], tf.float32), -1)
            inputs = inputs * mask

        ConvBankWithPool    = Conv1dBankWithMaxPool(self.bank_K)
        ConvProj            = Conv1dProjection(self.proj_unit)
        Highway             = FCHighwayNet(self.highway_layers)
        cell                = GRUCell(self.proj_unit[1])
        fw_cell             = FusedRNNCellAdaptor(cell)
        bw_cell             = TimeReversedFusedRNN(fw_cell)

        ### calculate
        # conv net
        output_0 = ConvBankWithPool(inputs, is_training)

        ### for correctness.
        if sequence_length is not None:
            output_0 = output_0 * mask

        output_1 = ConvProj(output_0, is_training)
        # residual connect
        res_output = tf.identity(inputs) + output_1

        # highway net
        highway_output = Highway(res_output)

        # biGRU
        # time major
        bGRUinp = tf.transpose(highway_output, perm=(1, 0, 2))
        fw_out, _ = fw_cell(bGRUinp, sequence_length=sequence_length, scope="fw", dtype=tf.float32)
        bw_out, _ = bw_cell(bGRUinp, sequence_length=sequence_length, scope="bw", dtype=tf.float32)
        final_output = tf.concat([fw_out, bw_out], axis=-1)

        if not time_major:
            final_output = tf.transpose(final_output, perm=(1,0,2))

        return final_output

