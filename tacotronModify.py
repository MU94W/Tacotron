import tensorflow as tf
from TFCommon.Model import Model
from Tacotron.Modules.CBHG import CBHG
from TFAttention.Attention import BahdanauAttentionModule as AttModule
from TFAttention.RNNCell import GRUCell, ResidualWrapper
from TFAttention.RNNDecoderCell import GRUDecoderCell
from TFAttention.metrics import binary_accuracy

class Tacotron(Model):
    """Tacotron
    """
    def __init__(self, r=5, is_training=True):
        super(Tacotron, self).__init__()
        self.__r = r
        self.__is_training = is_training

    @property
    def r(self):
        return self.__r

    @property
    def is_training(self):
        return self.__is_training

    def build(self, inputs, outputs, embed_class, time_major=None):
        assert time_major is not None, "[*] You must specify whether is time_major or not!"
        if time_major:
            input_time_steps = tf.shape(inputs)[0]
        else:
            input_time_steps = tf.shape(inputs)[1]

        ### Encoder ###
        ### Embedding
        with tf.variable_scope('encoder'):
            with tf.variable_scope('embed'):
                embed_w = tf.get_variable(name='W', shape=(embed_class, 256), dtype=tf.float32)
                embeded = tf.nn.embedding_lookup(embed_w, inputs)
            with tf.variable_scope('pre-net'):
                pre_0 = tf.layers.dropout(tf.layers.dense(embeded, 256, tf.nn.relu))
                pre_1 = tf.layers.dropout(tf.layers.dense(pre_0, 128, tf.nn.relu))
            with tf.variable_scope('CBHG'):
                cbhg_net = CBHG(16, (128, 128))
                cbhg_out = cbhg_net(pre_1, self.is_training, time_major)

        with tf.variable_scope('decoder'):
            with tf.variable_scope('attention'):
                att_module = AttModule(256, cbhg_out, time_major)
            with tf.variable_scope('init_att_state'):
                if time_major:
                    h_bw_1 = cbhg_out[0, :, 128:]
                else:
                    h_bw_1 = cbhg_out[:, 0, 128:]
                att_init = tf.layers.dense(h_bw_1, 256, tf.nn.tanh)
            att_rnn = GRUDecoderCell(256, att_init)
            dec_rnn_0 = GRUDecoderCell(256)
            dec_rnn_1 = ResidualWrapper(GRUCell(256))
            ### prepare loop
            with tf.variable_scope('loop'):
                if not time_major:
                    outputs = tf.transpose(outputs, perm=(1,0,2))
                max_time_steps = tf.shape(outputs)[0]
                reduced_time_steps = tf.div(max_time_steps - 1, self.r) + 1
                batch_size = tf.shape(outputs)[1]
                output_dim = outputs.get_shape()[-1].value
                att_rnn_state   = att_rnn.init_state(batch_size, tf.float32)
                dec_rnn_0_state = dec_rnn_0.init_state(batch_size, tf.float32)
                dec_rnn_1_state = dec_rnn_1.init_state(batch_size, tf.float32)
                state_tup = tuple([att_rnn_state, dec_rnn_0_state, dec_rnn_1_state])
                ### prepare tensor array
                output_ta = tf.TensorArray(size=reduced_time_steps * self.r, dtype=tf.float32)
                alpha_ta  = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)

                time = tf.constant(0, dtype=tf.int32)
                cond = lambda time, *_: tf.less(time, reduced_time_steps)
                def body(time, output_ta, alpha_ta, state_tup):
                    begin_step = time*self.r
                    ### get indication
                    this_indic = state_tup[2][0]
                    ### pre-net
                    with tf.variable_scope('pre-net'):
                        pre_0 = tf.layers.dropout(tf.layers.dense(this_indic, 256, tf.nn.relu))
                        pre_1 = tf.layers.dropout(tf.layers.dense(pre_0, 128, tf.nn.relu))
                    with tf.variable_scope('attention'):
                        query = state_tup[0][0]
                        context, alpha = att_module(query)
                        alpha_ta = alpha_ta.write(time, alpha)
                    with tf.variable_scope('att-rnn'):
                        output_att, att_rnn_state = att_rnn(pre_1, state_tup[0], context)
                    with tf.variable_scope('decoder-rnn'):
                        with tf.variable_scope('cell-0'):
                            output_0, dec_rnn_0_state = dec_rnn_0(output_att, state_tup[1], context)
                            res_output_0 = tf.identity(query) + output_0
                        with tf.variable_scope('cell-1'):
                            res_output_1, dec_rnn_1_state = dec_rnn_1(res_output_0, state_tup[2])
                    with tf.variable_scope('expand-dense'):
                        dense_out_mgc_lf0 = tf.layers.dense(res_output_1, output_dim-1)
                        dense_out_vuv = tf.layers.dense(res_output_1, 1, tf.sigmoid)
                        dense_out = tf.concat([dense_out_mgc_lf0, dense_out_vuv], axis=-1)
                        for idr in range(self.r):
                            output_ta = output_ta.write(begin_step + idr, dense_out)

                    state_tup = tuple([att_rnn_state, dec_rnn_0_state, dec_rnn_1_state])
                    return tf.add(time, 1), output_ta, alpha_ta, state_tup

                ### run loop
                _, final_output_ta, final_alpha_ta, *_ = tf.while_loop(cond, body, [time, output_ta, alpha_ta, state_tup])

            ### time major
            final_output = tf.reshape(final_output_ta.stack(), shape=(reduced_time_steps * self.r, batch_size, output_dim))
            final_output = final_output[:max_time_steps]
            final_alpha  = tf.reshape(final_alpha_ta.stack(),  shape=(reduced_time_steps, batch_size, input_time_steps))

        self.loss_mgc_lf0 = tf.losses.mean_squared_error(outputs[:, :, :-1], final_output[:, :, :-1])
        self.loss_vuv = tf.losses.sigmoid_cross_entropy(outputs[:, :, -1], final_output[:, :, -1])
        self.loss = 0.9*self.loss_mgc_lf0 + 0.1*self.loss_vuv
        self.metric_vuv = binary_accuracy(outputs[:, :, -1], final_output[:, :, -1])
        self.alpha = final_alpha

    def summary(self, suffix):
        sum_0 = tf.summary.scalar('%s/loss_mgc_lf0' % suffix, self.loss_mgc_lf0)
        sum_1 = tf.summary.scalar('%s/loss_vuv' % suffix, self.loss_vuv)
        sum_2 = tf.summary.scalar('%s/loss' % suffix, self.loss)
        sum_3 = tf.summary.scalar('%s/metric_vuv_acc' % suffix, self.metric_vuv)
        ### prepare alpha img
        ob_alpha = self.alpha[:,:2]
        ob_alpha = tf.transpose(ob_alpha, perm=(1,0,2))     # batch major
        out_steps = tf.shape(ob_alpha)[1]
        inp_steps = tf.shape(ob_alpha)[2]
        ob_alpha_img = tf.reshape(ob_alpha, shape=(2, out_steps, inp_steps, 1))
        sum_4 = tf.summary.image('%s/alpha' % suffix, ob_alpha_img)
        
        return tf.summary.merge([sum_0, sum_1, sum_2, sum_3, sum_4])


class mTacotron(Tacotron):
    """modified Tacotron
    """
    def build(self, inputs, outputs, time_major=None):
        assert time_major is not None, "[*] You must specify whether is time_major or not!"
        if time_major:
            input_time_steps = tf.shape(inputs)[0]
        else:
            input_time_steps = tf.shape(inputs)[1]

        ### Encoder ###
        ### Embedding
        with tf.variable_scope('encoder'):
            with tf.variable_scope('bottom-net'):
                bottom_rep = tf.layers.dense(inputs, 256, tf.nn.relu)
            with tf.variable_scope('pre-net'):
                pre_0 = tf.layers.dropout(tf.layers.dense(bottom_rep, 256, tf.nn.relu))
                pre_1 = tf.layers.dropout(tf.layers.dense(pre_0, 128, tf.nn.relu))
            with tf.variable_scope('CBHG'):
                cbhg_net = CBHG(16, (128, 128))
                cbhg_out = cbhg_net(pre_1, self.is_training, time_major)

        with tf.variable_scope('decoder'):
            with tf.variable_scope('attention'):
                att_module = AttModule(256, cbhg_out, time_major)
            att_rnn = GRUCell(256)
            dec_rnn_0 = GRUDecoderCell(256)
            dec_rnn_1 = ResidualWrapper(GRUCell(256))
            ### prepare loop
            with tf.variable_scope('loop'):
                if not time_major:
                    outputs = tf.transpose(outputs, perm=(1,0,2))
                max_time_steps = tf.shape(outputs)[0]
                reduced_time_steps = tf.div(max_time_steps, self.r)
                batch_size = tf.shape(outputs)[1]
                output_dim = outputs.get_shape()[-1].value
                pad_indic = tf.zeros(shape=(self.r, batch_size, output_dim), dtype=tf.float32)
                indic = tf.concat([pad_indic, outputs], axis=0)
                att_rnn_state   = att_rnn.init_state(batch_size, tf.float32)
                dec_rnn_0_state = dec_rnn_0.init_state(batch_size, tf.float32)
                dec_rnn_1_state = dec_rnn_1.init_state(batch_size, tf.float32)
                state_tup = tuple([att_rnn_state, dec_rnn_0_state, dec_rnn_1_state])
                ### prepare tensor array
                output_ta = tf.TensorArray(size=max_time_steps, dtype=tf.float32)
                alpha_ta  = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)

                time = tf.constant(0, dtype=tf.int32)
                cond = lambda time, *_: tf.less(time, reduced_time_steps)
                def body(time, output_ta, alpha_ta, state_tup):
                    begin_step = time*self.r
                    ### get indication
                    this_indic = indic[begin_step + self.r - 1]     # 只用最后一帧
                    ### pre-net
                    with tf.variable_scope('pre-net'):
                        pre_0 = tf.layers.dropout(tf.layers.dense(this_indic, 256, tf.nn.relu))
                        pre_1 = tf.layers.dropout(tf.layers.dense(pre_0, 128, tf.nn.relu))
                    with tf.variable_scope('att-rnn'):
                        query, att_rnn_state = att_rnn(pre_1, state_tup[0])
                    with tf.variable_scope('attention'):
                        context, alpha = att_module(query)
                        alpha_ta = alpha_ta.write(time, alpha)
                    with tf.variable_scope('decoder-rnn'):
                        with tf.variable_scope('cell-0'):
                            output_0, dec_rnn_0_state = dec_rnn_0(query, state_tup[1], context)
                            res_output_0 = tf.identity(query) + output_0
                        with tf.variable_scope('cell-1'):
                            res_output_1, dec_rnn_1_state = dec_rnn_1(res_output_0, state_tup[2])
                    with tf.variable_scope('expand-dense'):
                        dense_out = tf.layers.dense(res_output_1, output_dim)
                        for idr in range(self.r):
                            output_ta = output_ta.write(begin_step + idr, dense_out)

                    state_tup = tuple([att_rnn_state, dec_rnn_0_state, dec_rnn_1_state])
                    return tf.add(time, 1), output_ta, alpha_ta, state_tup

                ### run loop
                _, final_output_ta, final_alpha_ta, *_ = tf.while_loop(cond, body, [time, output_ta, alpha_ta, state_tup])

            final_output = tf.reshape(final_output_ta.stack(), shape=(max_time_steps, batch_size, output_dim))
            final_alpha  = tf.reshape(final_alpha_ta.stack(),  shape=(reduced_time_steps, batch_size, input_time_steps))

        self.loss = tf.losses.mean_squared_error(outputs, final_output)
        self.alpha = final_alpha

