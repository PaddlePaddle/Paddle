import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, BatchNorm, Embedding, GRUUnit

from text import DynamicDecode, RNN, BasicLSTMCell, RNNCell
from model import Model, Loss


class ConvBNPool(fluid.dygraph.Layer):
    def __init__(self,
                 out_ch,
                 channels,
                 act="relu",
                 is_test=False,
                 pool=True,
                 use_cudnn=True):
        super(ConvBNPool, self).__init__()
        self.pool = pool

        filter_size = 3
        conv_std_0 = (2.0 / (filter_size**2 * channels[0]))**0.5
        conv_param_0 = fluid.ParamAttr(
            initializer=fluid.initializer.Normal(0.0, conv_std_0))

        conv_std_1 = (2.0 / (filter_size**2 * channels[1]))**0.5
        conv_param_1 = fluid.ParamAttr(
            initializer=fluid.initializer.Normal(0.0, conv_std_1))

        self.conv_0_layer = Conv2D(
            channels[0],
            out_ch[0],
            3,
            padding=1,
            param_attr=conv_param_0,
            bias_attr=False,
            act=None,
            use_cudnn=use_cudnn)
        self.bn_0_layer = BatchNorm(out_ch[0], act=act, is_test=is_test)
        self.conv_1_layer = Conv2D(
            out_ch[0],
            num_filters=out_ch[1],
            filter_size=3,
            padding=1,
            param_attr=conv_param_1,
            bias_attr=False,
            act=None,
            use_cudnn=use_cudnn)
        self.bn_1_layer = BatchNorm(out_ch[1], act=act, is_test=is_test)

        if self.pool:
            self.pool_layer = Pool2D(
                pool_size=2,
                pool_type='max',
                pool_stride=2,
                use_cudnn=use_cudnn,
                ceil_mode=True)

    def forward(self, inputs):
        conv_0 = self.conv_0_layer(inputs)
        bn_0 = self.bn_0_layer(conv_0)
        conv_1 = self.conv_1_layer(bn_0)
        bn_1 = self.bn_1_layer(conv_1)
        if self.pool:
            bn_pool = self.pool_layer(bn_1)

            return bn_pool
        return bn_1


class OCRConv(fluid.dygraph.Layer):
    def __init__(self, is_test=False, use_cudnn=True):
        super(OCRConv, self).__init__()
        self.conv_bn_pool_1 = ConvBNPool(
            [16, 16], [1, 16], is_test=is_test, use_cudnn=use_cudnn)
        self.conv_bn_pool_2 = ConvBNPool(
            [32, 32], [16, 32], is_test=is_test, use_cudnn=use_cudnn)
        self.conv_bn_pool_3 = ConvBNPool(
            [64, 64], [32, 64], is_test=is_test, use_cudnn=use_cudnn)
        self.conv_bn_pool_4 = ConvBNPool(
            [128, 128], [64, 128],
            is_test=is_test,
            pool=False,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        inputs_1 = self.conv_bn_pool_1(inputs)
        inputs_2 = self.conv_bn_pool_2(inputs_1)
        inputs_3 = self.conv_bn_pool_3(inputs_2)
        inputs_4 = self.conv_bn_pool_4(inputs_3)

        return inputs_4


class GRUCell(RNNCell):
    def __init__(self,
                 size,
                 param_attr=None,
                 bias_attr=None,
                 is_reverse=False,
                 gate_activation='sigmoid',
                 candidate_activation='tanh',
                 origin_mode=False,
                 init_size=None):
        super(GRUCell, self).__init__()

        self.input_proj = Linear(
            768, size * 3, param_attr=param_attr, bias_attr=False)

        self.gru_unit = GRUUnit(
            size * 3,
            param_attr=param_attr,
            bias_attr=bias_attr,
            activation=candidate_activation,
            gate_activation=gate_activation,
            origin_mode=origin_mode)

        self.size = size
        self.is_reverse = is_reverse

    def forward(self, inputs, states):
        # step_outputs, new_states = cell(step_inputs, states)
        # for GRUCell, `step_outputs` and `new_states` both are hidden
        x = self.input_proj(inputs)
        hidden, _, _ = self.gru_unit(x, states)
        return hidden, hidden


class DecoderCell(RNNCell):
    def __init__(self, size):
        self.gru = GRUCell(size)
        self.attention = SimpleAttention(size)
        self.fc_1_layer = Linear(
            input_dim=size * 2, output_dim=size * 3, bias_attr=False)
        self.fc_2_layer = Linear(
            input_dim=size, output_dim=size * 3, bias_attr=False)

    def forward(self, inputs, states, encoder_vec, encoder_proj):
        context = self.attention(encoder_vec, encoder_proj, states)
        fc_1 = self.fc_1_layer(context)
        fc_2 = self.fc_2_layer(inputs)
        decoder_inputs = fluid.layers.elementwise_add(x=fc_1, y=fc_2)
        h, _ = self.gru(decoder_inputs, states)
        return h, h


class Decoder(fluid.dygraph.Layer):
    def __init__(self, size, num_classes):
        super(Decoder, self).__init__()
        self.embedder = Embedding(size=[num_classes, size])
        self.gru_attention = RNN(DecoderCell(size),
                                 is_reverse=False,
                                 time_major=False)
        self.output_layer = Linear(size, num_classes, bias_attr=False)

    def forward(self, target, decoder_initial_states, encoder_vec,
                encoder_proj):
        inputs = self.embedder(target)
        decoder_output, _ = self.gru_attention(
            inputs,
            initial_states=decoder_initial_states,
            encoder_vec=encoder_vec,
            encoder_proj=encoder_proj)
        predict = self.output_layer(decoder_output)
        return predict


class EncoderNet(fluid.dygraph.Layer):
    def __init__(self,
                 batch_size,
                 decoder_size,
                 rnn_hidden_size=200,
                 is_test=False,
                 use_cudnn=True):
        super(EncoderNet, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size
        para_attr = fluid.ParamAttr(
            initializer=fluid.initializer.Normal(0.0, 0.02))
        bias_attr = fluid.ParamAttr(
            initializer=fluid.initializer.Normal(0.0, 0.02), learning_rate=2.0)
        self.ocr_convs = OCRConv(is_test=is_test, use_cudnn=use_cudnn)

        self.fc_1_layer = Linear(
            768, rnn_hidden_size * 3, param_attr=para_attr, bias_attr=False)
        self.fc_2_layer = Linear(
            768, rnn_hidden_size * 3, param_attr=para_attr, bias_attr=False)
        self.gru_forward_layer = DynamicGRU(
            size=rnn_hidden_size,
            param_attr=para_attr,
            bias_attr=bias_attr,
            candidate_activation='relu')
        self.gru_backward_layer = DynamicGRU(
            size=rnn_hidden_size,
            param_attr=para_attr,
            bias_attr=bias_attr,
            candidate_activation='relu',
            is_reverse=True)

        self.encoded_proj_fc = Linear(
            rnn_hidden_size * 2, decoder_size, bias_attr=False)

    def forward(self, inputs):
        conv_features = self.ocr_convs(inputs)
        transpose_conv_features = fluid.layers.transpose(
            conv_features, perm=[0, 3, 1, 2])

        sliced_feature = fluid.layers.reshape(
            transpose_conv_features, [
                -1, transpose_conv_features.shape[1],
                transpose_conv_features.shape[2] *
                transpose_conv_features.shape[3]
            ],
            inplace=False)

        fc_1 = self.fc_1_layer(sliced_feature)

        fc_2 = self.fc_2_layer(sliced_feature)

        gru_forward = self.gru_forward_layer(fc_1)

        gru_backward = self.gru_backward_layer(fc_2)

        encoded_vector = fluid.layers.concat(
            input=[gru_forward, gru_backward], axis=2)

        encoded_proj = self.encoded_proj_fc(encoded_vector)

        return gru_backward, encoded_vector, encoded_proj


class SimpleAttention(fluid.dygraph.Layer):
    def __init__(self, decoder_size):
        super(SimpleAttention, self).__init__()

        self.fc_1 = Linear(
            decoder_size, decoder_size, act=None, bias_attr=False)
        self.fc_2 = Linear(decoder_size, 1, act=None, bias_attr=False)

    def forward(self, encoder_vec, encoder_proj, decoder_state):

        decoder_state_fc = self.fc_1(decoder_state)

        decoder_state_proj_reshape = fluid.layers.reshape(
            decoder_state_fc, [-1, 1, decoder_state_fc.shape[1]],
            inplace=False)
        decoder_state_expand = fluid.layers.expand(
            decoder_state_proj_reshape, [1, encoder_proj.shape[1], 1])
        concated = fluid.layers.elementwise_add(encoder_proj,
                                                decoder_state_expand)
        concated = fluid.layers.tanh(x=concated)
        attention_weight = self.fc_2(concated)
        weights_reshape = fluid.layers.reshape(
            x=attention_weight, shape=[concated.shape[0], -1], inplace=False)

        weights_reshape = fluid.layers.softmax(weights_reshape)
        scaled = fluid.layers.elementwise_mul(
            x=encoder_vec, y=weights_reshape, axis=0)

        context = fluid.layers.reduce_sum(scaled, dim=1)

        return context


class GRUDecoderWithAttention(fluid.dygraph.Layer):
    def __init__(self, encoder_size, decoder_size, num_classes):
        super(GRUDecoderWithAttention, self).__init__()
        self.simple_attention = SimpleAttention(decoder_size)

        self.fc_1_layer = Linear(
            input_dim=encoder_size * 2,
            output_dim=decoder_size * 3,
            bias_attr=False)
        self.fc_2_layer = Linear(
            input_dim=decoder_size,
            output_dim=decoder_size * 3,
            bias_attr=False)
        self.gru_unit = GRUUnit(
            size=decoder_size * 3, param_attr=None, bias_attr=None)
        self.out_layer = Linear(
            input_dim=decoder_size,
            output_dim=num_classes + 2,
            bias_attr=None,
            act='softmax')

        self.decoder_size = decoder_size

    def forward(self,
                current_word,
                encoder_vec,
                encoder_proj,
                decoder_boot,
                inference=False):
        current_word = fluid.layers.reshape(
            current_word, [-1, current_word.shape[2]], inplace=False)

        context = self.simple_attention(encoder_vec, encoder_proj,
                                        decoder_boot)
        fc_1 = self.fc_1_layer(context)
        fc_2 = self.fc_2_layer(current_word)
        decoder_inputs = fluid.layers.elementwise_add(x=fc_1, y=fc_2)

        h, _, _ = self.gru_unit(decoder_inputs, decoder_boot)
        out = self.out_layer(h)

        return out, h


class OCRAttention(fluid.dygraph.Layer):
    def __init__(self, batch_size, num_classes, encoder_size, decoder_size,
                 word_vector_dim):
        super(OCRAttention, self).__init__()
        self.encoder_net = EncoderNet(batch_size, decoder_size)
        self.fc = Linear(
            input_dim=encoder_size,
            output_dim=decoder_size,
            bias_attr=False,
            act='relu')
        self.embedding = Embedding(
            [num_classes + 2, word_vector_dim], dtype='float32')
        self.gru_decoder_with_attention = GRUDecoderWithAttention(
            encoder_size, decoder_size, num_classes)
        self.batch_size = batch_size

    def forward(self, inputs, label_in):
        gru_backward, encoded_vector, encoded_proj = self.encoder_net(inputs)
        backward_first = fluid.layers.slice(
            gru_backward, axes=[1], starts=[0], ends=[1])
        backward_first = fluid.layers.reshape(
            backward_first, [-1, backward_first.shape[2]], inplace=False)

        decoder_boot = self.fc(backward_first)

        label_in = fluid.layers.reshape(label_in, [-1], inplace=False)
        trg_embedding = self.embedding(label_in)

        trg_embedding = fluid.layers.reshape(
            trg_embedding, [self.batch_size, -1, trg_embedding.shape[1]],
            inplace=False)

        pred_temp = []
        for i in range(trg_embedding.shape[1]):
            current_word = fluid.layers.slice(
                trg_embedding, axes=[1], starts=[i], ends=[i + 1])
            out, decoder_boot = self.gru_decoder_with_attention(
                current_word, encoded_vector, encoded_proj, decoder_boot)
            pred_temp.append(out)
        pred_temp = fluid.layers.concat(pred_temp, axis=1)

        batch_size = trg_embedding.shape[0]
        seq_len = trg_embedding.shape[1]
        prediction = fluid.layers.reshape(
            pred_temp, shape=[batch_size, seq_len, -1])

        return prediction
