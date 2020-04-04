import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, BatchNorm, Embedding, GRUUnit

from text import DynamicDecode, RNN, RNNCell
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


class SimpleAttention(fluid.dygraph.Layer):
    def __init__(self, decoder_size):
        super(SimpleAttention, self).__init__()

        self.fc1 = Linear(decoder_size, decoder_size, bias_attr=False)
        self.fc2 = Linear(decoder_size, 1, bias_attr=False)

    def forward(self, encoder_vec, encoder_proj, decoder_state):
        decoder_state = self.fc1(decoder_state)
        decoder_state = fluid.layers.unsqueeze(decoder_state, [1])

        mix = fluid.layers.elementwise_add(encoder_proj, decoder_state)
        mix = fluid.layers.tanh(x=mix)

        attn_score = self.fc2(mix)
        attn_scores = layers.squeeze(attn_score, [2])
        attn_scores = fluid.layers.softmax(attn_scores)

        scaled = fluid.layers.elementwise_mul(
            x=encoder_vec, y=attn_scores, axis=0)

        context = fluid.layers.reduce_sum(scaled, dim=1)
        return context


class GRUCell(RNNCell):
    def __init__(self,
                 input_size,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation='sigmoid',
                 candidate_activation='tanh',
                 origin_mode=False):
        super(GRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.fc_layer = Linear(
            input_size,
            hidden_size * 3,
            param_attr=param_attr,
            bias_attr=False)

        self.gru_unit = GRUUnit(
            hidden_size * 3,
            param_attr=param_attr,
            bias_attr=bias_attr,
            activation=candidate_activation,
            gate_activation=gate_activation,
            origin_mode=origin_mode)

    def forward(self, inputs, states):
        # step_outputs, new_states = cell(step_inputs, states)
        # for GRUCell, `step_outputs` and `new_states` both are hidden
        x = self.fc_layer(inputs)
        hidden, _, _ = self.gru_unit(x, states)
        return hidden, hidden

    @property
    def state_shape(self):
        return [self.hidden_size]


class EncoderNet(fluid.dygraph.Layer):
    def __init__(self,
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

        self.gru_forward_layer = RNN(
            cell=GRUCell(
                input_size=128 * 6,  # channel * h
                hidden_size=rnn_hidden_size,
                param_attr=para_attr,
                bias_attr=bias_attr,
                candidate_activation='relu'),
            is_reverse=False,
            time_major=False)
        self.gru_backward_layer = RNN(
            cell=GRUCell(
                input_size=128 * 6,  # channel * h
                hidden_size=rnn_hidden_size,
                param_attr=para_attr,
                bias_attr=bias_attr,
                candidate_activation='relu'),
            is_reverse=True,
            time_major=False)

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

        gru_forward, _ = self.gru_forward_layer(sliced_feature)

        gru_backward, _ = self.gru_backward_layer(sliced_feature)

        encoded_vector = fluid.layers.concat(
            input=[gru_forward, gru_backward], axis=2)

        encoded_proj = self.encoded_proj_fc(encoded_vector)

        return gru_backward, encoded_vector, encoded_proj


class DecoderCell(RNNCell):
    def __init__(self, encoder_size, decoder_size):
        super(DecoderCell, self).__init__()
        self.attention = SimpleAttention(decoder_size)
        self.gru_cell = GRUCell(
            input_size=encoder_size * 2 +
            decoder_size,  # encoded_vector.shape[-1] + embed_size
            hidden_size=decoder_size)

    def forward(self, current_word, states, encoder_vec, encoder_proj):
        context = self.attention(encoder_vec, encoder_proj, states)
        decoder_inputs = layers.concat([current_word, context], axis=1)
        hidden, _ = self.gru_cell(decoder_inputs, states)
        return hidden, hidden


class GRUDecoderWithAttention(fluid.dygraph.Layer):
    def __init__(self, encoder_size, decoder_size, num_classes):
        super(GRUDecoderWithAttention, self).__init__()
        self.gru_attention = RNN(DecoderCell(encoder_size, decoder_size),
                                 is_reverse=False,
                                 time_major=False)
        self.out_layer = Linear(
            input_dim=decoder_size,
            output_dim=num_classes + 2,
            bias_attr=None,
            act='softmax')

    def forward(self, inputs, decoder_initial_states, encoder_vec,
                encoder_proj):
        out, _ = self.gru_attention(
            inputs,
            initial_states=decoder_initial_states,
            encoder_vec=encoder_vec,
            encoder_proj=encoder_proj)
        predict = self.out_layer(out)
        return predict


class OCRAttention(Model):
    def __init__(self, num_classes, encoder_size, decoder_size,
                 word_vector_dim):
        super(OCRAttention, self).__init__()
        self.encoder_net = EncoderNet(decoder_size)
        self.fc = Linear(
            input_dim=encoder_size,
            output_dim=decoder_size,
            bias_attr=False,
            act='relu')
        self.embedding = Embedding(
            [num_classes + 2, word_vector_dim], dtype='float32')
        self.gru_decoder_with_attention = GRUDecoderWithAttention(
            encoder_size, decoder_size, num_classes)

    def forward(self, inputs, label_in):
        gru_backward, encoded_vector, encoded_proj = self.encoder_net(inputs)

        decoder_boot = self.fc(gru_backward[:, 0])
        trg_embedding = self.embedding(label_in)
        prediction = self.gru_decoder_with_attention(
            trg_embedding, decoder_boot, encoded_vector, encoded_proj)

        return prediction


class CrossEntropyCriterion(Loss):
    def __init__(self):
        super(CrossEntropyCriterion, self).__init__()

    def forward(self, outputs, labels):
        predict, (label, mask) = outputs[0], labels

        loss = layers.cross_entropy(predict, label=label, soft_label=False)
        loss = layers.elementwise_mul(loss, mask, axis=0)
        loss = layers.reduce_sum(loss)
        return loss
