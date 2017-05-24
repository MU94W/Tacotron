import tensorflow as tf
from TFCommon.RNNCell import GRUCell
from TFCommon.DynamicRNNScan import biDynamicRNNScan
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

    def __call__(self, inputs, is_training=True, time_major=None):
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
        output_0 = ConvBankWithPool(inputs, is_training)
        output_1 = ConvProj(output_0, is_training)
        # residual connect
        res_output = tf.identity(inputs) + output_1

        # highway net
        highway_output = Highway(res_output)
        highway_output = tf.transpose(highway_output, perm=(1,0,2))

        # biGRU
        final_output, *_ = biDynamicRNNScan(rnn_cell_fw, rnn_cell_bw, highway_output)
        if not time_major:
            final_output = tf.transpose(final_output, perm=(1,0,2))

        return final_output

