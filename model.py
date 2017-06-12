import tensorflow as tf
from TFCommon.Model import Model
from Tacotron.Modules.CBHG import CBHG
from TFCommon.Attention import BahdanauAttentionModule as AttModule
from TFCommon.RNNCell import GRUCell, ResidualWrapper
from TFCommon.Layers import EmbeddingLayer

class Tacotron(Model):
    """Tacotron
    """
    def __init__(self, r=2, lambda_l1=0., is_training=True):
        super(Tacotron, self).__init__()
        self.__r = r
        self.lambda_l1 = lambda_l1
        self.__is_training = is_training

    @property
    def r(self):
        return self.__r

    @property
    def is_training(self):
        return self.__is_training

    def build_forward(self, inputs, outputs, embed_class, time_major=None):
        assert time_major is not None, "[*] You must specify whether is time_major or not!"
        if time_major:
            input_time_steps = tf.shape(inputs)[0]
        else:
            input_time_steps = tf.shape(inputs)[1]

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step = global_step

        ### Encoder ###
        ### Embedding
        with tf.variable_scope('encoder'):
            embeded = EmbeddingLayer(embed_class, 256)(inputs, scope='chr-emb')
            with tf.variable_scope('pre-net'):
                pre_0 = tf.layers.dropout(tf.layers.dense(embeded, 256, tf.nn.relu))
                pre_1 = tf.layers.dropout(tf.layers.dense(pre_0, 128, tf.nn.relu))
            with tf.variable_scope('CBHG'):
                cbhg_net = CBHG(16, (128, 128))
                cbhg_out = cbhg_net(pre_1, self.is_training, time_major)

        with tf.variable_scope('decoder'):
            with tf.variable_scope('att-memory'):
                att_module = AttModule(256, cbhg_out, time_major=time_major)
            att_rnn = GRUCell(256)
            dec_rnn_0 = GRUCell(256)
            dec_rnn_1 = ResidualWrapper(GRUCell(256))
            ### prepare loop
            with tf.variable_scope('loop'):
                if not time_major:
                    outputs = tf.transpose(outputs, perm=(1,0,2))
                max_time_steps = tf.shape(outputs)[0]
                reduced_time_steps = tf.div(max_time_steps - 1, self.r) + 1
                batch_size = tf.shape(outputs)[1]
                output_dim = outputs.shape[-1].value
                pad_indic = tf.zeros(shape=(self.r, batch_size, output_dim), dtype=tf.float32)
                indic = tf.concat([pad_indic, outputs], axis=0)
                pred_indic = tf.zeros(shape=(batch_size, output_dim), dtype=tf.float32)
                att_rnn_state   = att_rnn.init_state(batch_size, tf.float32)
                dec_rnn_0_state = dec_rnn_0.init_state(batch_size, tf.float32)
                dec_rnn_1_state = dec_rnn_1.init_state(batch_size, tf.float32)
                state_tup = tuple([att_rnn_state, dec_rnn_0_state, dec_rnn_1_state])
                ### prepare tensor array
                output_ta = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)
                alpha_ta  = tf.TensorArray(size=reduced_time_steps, dtype=tf.float32)

                time = tf.constant(0, dtype=tf.int32)
                cond = lambda time, *_: tf.less(time, reduced_time_steps)
                def body(time, pred_indic, output_ta, alpha_ta, state_tup):
                    ### get indication
                    if self.is_training:
                        this_indic = indic[self.r * time]     # 只用最后一帧
                    else:
                        this_indic = pred_indic
                    ### pre-net
                    with tf.variable_scope('pre-net'):
                        pre_0 = tf.layers.dropout(tf.layers.dense(this_indic, 256, tf.nn.relu))
                        pre_1 = tf.layers.dropout(tf.layers.dense(pre_0, 128, tf.nn.relu))
                    with tf.variable_scope('att-rnn'):
                        att_rnn_out, att_rnn_state = att_rnn(pre_1, state_tup[0])
                    with tf.variable_scope('att-query'):
                        query = att_rnn_out
                        context, alpha = att_module(query)
                        alpha_ta = alpha_ta.write(time, alpha)
                    with tf.variable_scope('decoder-rnn'):
                        with tf.variable_scope('cell-0'):
                            dec_rnn_0_inp = tf.concat([context, att_rnn_out], axis=-1)
                            dec_rnn_0_out, dec_rnn_0_state = dec_rnn_0(dec_rnn_0_inp, state_tup[1])
                        with tf.variable_scope('cell-1'):
                            dec_rnn_1_out, dec_rnn_1_state = dec_rnn_1(dec_rnn_0_out, state_tup[2])
                    with tf.variable_scope('dense-out'):
                        dense_out = tf.layers.dense(dec_rnn_1_out, output_dim)
                        out_mgc_lf0 = tf.reshape(\
                                tf.layers.dense(dec_rnn_1_out, (output_dim - 1) * self.r),
                                shape=(batch_size, self.r, output_dim - 1))
                        out_vuv = tf.reshape(\
                                tf.layers.dense(dec_rnn_1_out, self.r, tf.sigmoid),
                                shape=(batch_size, self.r, 1))
                        dense_out = tf.concat([out_mgc_lf0, out_vuv], axis=-1)
                        output_ta = output_ta.write(time, dense_out)
                    if not self.is_training:
                        mgc_lf0_indic = out_mgc_lf0[:, -1]
                        vuv_indic = tf.round(out_vuv[:, -1])
                        pred_indic = tf.concat([mgc_lf0_indic, vuv_indic], axis=-1)

                    state_tup = tuple([att_rnn_state, dec_rnn_0_state, dec_rnn_1_state])
                    return tf.add(time, 1), pred_indic, output_ta, alpha_ta, state_tup

                ### run loop
                _, _, final_output_ta, final_alpha_ta, *_ = tf.while_loop(cond, body, [time, pred_indic, output_ta, alpha_ta, state_tup])

            final_output = tf.reshape(final_output_ta.stack(), shape=(reduced_time_steps, batch_size, self.r, output_dim))
            final_output = tf.reshape(tf.transpose(final_output, perm=(0, 2, 1, 3)), shape=(reduced_time_steps * self.r, batch_size, output_dim))
            final_output = final_output[:max_time_steps]    # time major
            final_alpha = tf.reshape(final_alpha_ta.stack(),  shape=(reduced_time_steps, batch_size, input_time_steps))
            final_alpha = tf.transpose(final_alpha, perm=(1, 0, 2))     # batch major


        self.pred_out = final_output
        self.alpha_img = tf.expand_dims(final_alpha, -1)

        self.loss_mgc_lf0 = tf.losses.mean_squared_error(outputs[:, :, :-1], final_output[:, :, :-1])
        self.loss_vuv = tf.losses.sigmoid_cross_entropy(outputs[:, :, -1], final_output[:, :, -1])
        l1_reg = tf.contrib.layers.l1_regularizer(self.lambda_l1)
        l1_loss_vars = [item for item in tf.trainable_variables() if "decoder-rnn" in item.name or "dense-out" in item.name]
        self.l1_loss = tf.contrib.layers.apply_regularization(l1_reg, l1_loss_vars)
        self.loss = self.loss_mgc_lf0 + self.loss_vuv + self.l1_loss

    def build_backprop(self):
        with tf.variable_scope("backprop"):
            self.lr_start = tf.constant(0.001)
            self.learning_rate = tf.Variable(self.lr_start, name='learning_rate', trainable=False)
            self.opt = tf.train.AdamOptimizer(self.learning_rate)
            self.upd = self.opt.minimize(self.loss, global_step=self.global_step)
            return self.upd, self.lr_schedule_op

    def lr_schedule_op(self):
        lr_stage_0 = self.lr_start
        lr_stage_1 = tf.constant(0.0005)
        lr_stage_2 = tf.constant(0.0003)
        lr_state_3 = tf.constant(0.0001)
        gate_0 = tf.constant(int(5e5), dtype=tf.int32)
        gate_1 = tf.constant(int(1e6), dtype=tf.int32)
        gate_2 = tf.constant(int(2e6), dtype=tf.int32)
        def f1(): return lr_stage_0
        def f2(): return lr_stage_1
        def f3(): return lr_stage_2
        def f4(): return lr_stage_3
        new_lr = case([(tf.less(self.global_step, gate_0), f1), (tf.less(self.global_step, gate_1), f2),\
                (tf.less(self.global_step, gate_2), f3)],
                default=f4, exclusive=False)
        return self.learning_rate.assign(new_lr)

    def summary(self, suffix, num_img=2):
        sums = []
        sums.append(tf.summary.scalar('%s/loss_mgc_lf0' % suffix, self.loss_mgc_lf0))
        sums.append(tf.summary.scalar('%s/loss_vuv' % suffix, self.loss_vuv))
        sums.append(tf.summary.scalar('%s/loss_l1' % suffix, self.l1_loss))
        sums.append(tf.summary.scalar('%s/loss' % suffix, self.loss))
        sums.append(tf.summary.image('%s/alpha' % suffix, self.alpha_img[:num_img]))
        merged = tf.summary.merge(sums)
        return merged


