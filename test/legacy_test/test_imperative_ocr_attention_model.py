# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from test_imperative_base import new_program_scope

import paddle
from paddle import base
from paddle.base import core
from paddle.base.dygraph.base import to_variable
from paddle.nn import BatchNorm, Linear


class Config:
    '''
    config for training
    '''

    # encoder rnn hidden_size
    encoder_size = 8
    # decoder size for decoder stage
    decoder_size = 8
    # size for word embedding
    word_vector_dim = 8
    # max length for label padding
    max_length = 3
    # optimizer setting
    LR = 1.0
    learning_rate_decay = None

    # batch size to train
    batch_size = 2
    # class number to classify
    num_classes = 64

    use_gpu = False
    # special label for start and end
    SOS = 0
    EOS = 1
    # settings for ctc data, not use in unittest
    DATA_DIR_NAME = "./dataset/ctc_data/data"
    TRAIN_DATA_DIR_NAME = "train_images"
    TRAIN_LIST_FILE_NAME = "train.list"

    # data shape for input image
    DATA_SHAPE = [1, 16, 64]


class ConvBNPool(paddle.nn.Layer):
    def __init__(
        self,
        group,
        out_ch,
        channels,
        act="relu",
        is_test=False,
        pool=True,
        use_cudnn=True,
    ):
        super().__init__()
        self.group = group
        self.pool = pool

        filter_size = 3
        conv_std_0 = (2.0 / (filter_size**2 * channels[0])) ** 0.5
        conv_param_0 = base.ParamAttr(
            initializer=paddle.nn.initializer.Normal(0.0, conv_std_0)
        )

        conv_std_1 = (2.0 / (filter_size**2 * channels[1])) ** 0.5
        conv_param_1 = base.ParamAttr(
            initializer=paddle.nn.initializer.Normal(0.0, conv_std_1)
        )

        self.conv_0_layer = paddle.nn.Conv2D(
            channels[0],
            out_ch[0],
            3,
            padding=1,
            weight_attr=conv_param_0,
            bias_attr=False,
        )
        self.bn_0_layer = BatchNorm(out_ch[0], act=act, is_test=is_test)
        self.conv_1_layer = paddle.nn.Conv2D(
            out_ch[0],
            out_ch[1],
            3,
            padding=1,
            weight_attr=conv_param_1,
            bias_attr=False,
        )
        self.bn_1_layer = BatchNorm(out_ch[1], act=act, is_test=is_test)

        if self.pool:
            self.pool_layer = paddle.nn.MaxPool2D(
                kernel_size=2,
                stride=2,
                ceil_mode=True,
            )

    def forward(self, inputs):
        conv_0 = self.conv_0_layer(inputs)
        bn_0 = self.bn_0_layer(conv_0)
        conv_1 = self.conv_1_layer(bn_0)
        bn_1 = self.bn_1_layer(conv_1)
        if self.pool:
            bn_pool = self.pool_layer(bn_1)
            return bn_pool
        return bn_1


class OCRConv(paddle.nn.Layer):
    def __init__(self, is_test=False, use_cudnn=True):
        super().__init__()
        self.conv_bn_pool_1 = ConvBNPool(
            2, [8, 8], [1, 8], is_test=is_test, use_cudnn=use_cudnn
        )
        self.conv_bn_pool_2 = ConvBNPool(
            2, [8, 8], [8, 8], is_test=is_test, use_cudnn=use_cudnn
        )
        self.conv_bn_pool_3 = ConvBNPool(
            2, [8, 8], [8, 8], is_test=is_test, use_cudnn=use_cudnn
        )
        self.conv_bn_pool_4 = ConvBNPool(
            2,
            [16, 16],
            [8, 16],
            is_test=is_test,
            pool=False,
            use_cudnn=use_cudnn,
        )

    def forward(self, inputs):
        inputs_1 = self.conv_bn_pool_1(inputs)
        inputs_2 = self.conv_bn_pool_2(inputs_1)
        inputs_3 = self.conv_bn_pool_3(inputs_2)
        inputs_4 = self.conv_bn_pool_4(inputs_3)

        return inputs_4


class DynamicGRU(paddle.nn.Layer):
    def __init__(
        self,
        size,
        param_attr=None,
        bias_attr=None,
        is_reverse=False,
        gate_activation='sigmoid',
        candidate_activation='tanh',
        h_0=None,
        origin_mode=False,
    ):
        super().__init__()

        self.gru_unit = paddle.nn.GRUCell(
            size * 3,
            size,
        )

        self.h_0 = h_0
        self.is_reverse = is_reverse
        self.size = size

    def forward(self, inputs):
        hidden = self.h_0
        res = []
        for i in range(inputs.shape[1]):
            if self.is_reverse:
                i = inputs.shape[1] - 1 - i
            input_ = paddle.slice(inputs, axes=[1], starts=[i], ends=[i + 1])
            input_ = paddle.reshape(input_, [-1, input_.shape[2]])
            hidden, reset = self.gru_unit(input_, hidden)
            hidden_ = paddle.reshape(hidden, [-1, 1, hidden.shape[1]])
            if self.is_reverse:
                res = [hidden_] + res
            else:
                res.append(hidden_)
        res = paddle.concat(res, axis=1)
        return res


class EncoderNet(paddle.nn.Layer):
    def __init__(
        self, rnn_hidden_size=Config.encoder_size, is_test=False, use_cudnn=True
    ):
        super().__init__()
        self.rnn_hidden_size = rnn_hidden_size
        para_attr = base.ParamAttr(
            initializer=paddle.nn.initializer.Normal(0.0, 0.02)
        )
        bias_attr = base.ParamAttr(
            initializer=paddle.nn.initializer.Normal(0.0, 0.02),
            learning_rate=2.0,
        )
        if base.framework.in_dygraph_mode():
            h_0 = np.zeros(
                (Config.batch_size, rnn_hidden_size), dtype="float32"
            )
            h_0 = to_variable(h_0)
        else:
            h_0 = paddle.tensor.fill_constant(
                shape=[Config.batch_size, rnn_hidden_size],
                dtype='float32',
                value=0,
            )
        self.ocr_convs = OCRConv(is_test=is_test, use_cudnn=use_cudnn)

        self.fc_1_layer = Linear(
            32, rnn_hidden_size * 3, weight_attr=para_attr, bias_attr=False
        )
        self.fc_2_layer = Linear(
            32, rnn_hidden_size * 3, weight_attr=para_attr, bias_attr=False
        )
        self.gru_forward_layer = DynamicGRU(
            size=rnn_hidden_size,
            h_0=h_0,
            param_attr=para_attr,
            bias_attr=bias_attr,
            candidate_activation='relu',
        )
        self.gru_backward_layer = DynamicGRU(
            size=rnn_hidden_size,
            h_0=h_0,
            param_attr=para_attr,
            bias_attr=bias_attr,
            candidate_activation='relu',
            is_reverse=True,
        )

        self.encoded_proj_fc = Linear(
            rnn_hidden_size * 2, Config.decoder_size, bias_attr=False
        )

    def forward(self, inputs):
        conv_features = self.ocr_convs(inputs)
        # sliced_feature = base.layers.im2sequence(
        #    input=conv_features,
        #    stride=[1, 1],
        #    filter_size=[conv_features.shape[2], 1])

        transpose_conv_features = paddle.transpose(
            conv_features, perm=[0, 3, 1, 2]
        )
        sliced_feature = paddle.reshape(
            transpose_conv_features,
            [
                -1,
                8,
                transpose_conv_features.shape[2]
                * transpose_conv_features.shape[3],
            ],
        )
        fc_1 = self.fc_1_layer(sliced_feature)
        fc_2 = self.fc_2_layer(sliced_feature)
        gru_forward = self.gru_forward_layer(fc_1)

        gru_backward = self.gru_backward_layer(fc_2)

        encoded_vector = paddle.concat([gru_forward, gru_backward], axis=2)

        encoded_proj = self.encoded_proj_fc(encoded_vector)

        return gru_backward, encoded_vector, encoded_proj


class SimpleAttention(paddle.nn.Layer):
    def __init__(self, decoder_size):
        super().__init__()

        self.fc_1 = Linear(decoder_size, decoder_size, bias_attr=False)
        self.fc_2 = Linear(decoder_size, 1, bias_attr=False)

    def forward(self, encoder_vec, encoder_proj, decoder_state):
        decoder_state_fc = self.fc_1(decoder_state)
        decoder_state_proj_reshape = paddle.reshape(
            decoder_state_fc, [-1, 1, decoder_state_fc.shape[1]]
        )
        decoder_state_expand = paddle.expand(
            decoder_state_proj_reshape,
            [-1, encoder_proj.shape[1], -1],
        )
        concated = paddle.add(encoder_proj, decoder_state_expand)
        concated = paddle.tanh(x=concated)
        attention_weight = self.fc_2(concated)

        weights_reshape = paddle.reshape(
            x=attention_weight,
            shape=[attention_weight.shape[0], attention_weight.shape[1]],
        )

        weights_reshape = paddle.nn.functional.softmax(weights_reshape)
        scaled = paddle.tensor.math._multiply_with_axis(
            x=encoder_vec, y=weights_reshape, axis=0
        )
        context = paddle.sum(scaled, axis=1)

        return context


class GRUDecoderWithAttention(paddle.nn.Layer):
    def __init__(self, decoder_size, num_classes):
        super().__init__()
        self.simple_attention = SimpleAttention(decoder_size)

        self.fc_1_layer = Linear(
            Config.encoder_size * 2, decoder_size * 3, bias_attr=False
        )
        self.fc_2_layer = Linear(
            decoder_size, decoder_size * 3, bias_attr=False
        )
        self.gru_unit = paddle.nn.GRUCell(decoder_size * 3, decoder_size)
        self.out_layer = Linear(decoder_size, num_classes + 2, bias_attr=None)

        self.decoder_size = decoder_size

    def forward(
        self, target_embedding, encoder_vec, encoder_proj, decoder_boot
    ):
        res = []
        hidden_mem = decoder_boot
        for i in range(target_embedding.shape[1]):
            current_word = paddle.slice(
                target_embedding, axes=[1], starts=[i], ends=[i + 1]
            )
            current_word = paddle.reshape(
                current_word, [-1, current_word.shape[2]]
            )

            context = self.simple_attention(
                encoder_vec, encoder_proj, hidden_mem
            )
            fc_1 = self.fc_1_layer(context)
            fc_2 = self.fc_2_layer(current_word)
            decoder_inputs = paddle.add(x=fc_1, y=fc_2)

            h, _ = self.gru_unit(decoder_inputs, hidden_mem)
            hidden_mem = h
            out = self.out_layer(h)
            out = paddle.nn.functional.softmax(out)
            res.append(out)

        res1 = paddle.concat(res, axis=1)

        return res1


class OCRAttention(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.encoder_net = EncoderNet()
        self.fc = Linear(
            Config.encoder_size,
            Config.decoder_size,
            bias_attr=False,
        )
        self.embedding = paddle.nn.Embedding(
            Config.num_classes + 2, Config.word_vector_dim
        )
        self.gru_decoder_with_attention = GRUDecoderWithAttention(
            Config.decoder_size, Config.num_classes
        )

    def forward(self, inputs, label_in):
        gru_backward, encoded_vector, encoded_proj = self.encoder_net(inputs)
        backward_first = paddle.slice(
            gru_backward, axes=[1], starts=[0], ends=[1]
        )
        backward_first = paddle.reshape(
            backward_first, [-1, backward_first.shape[2]]
        )
        decoder_boot = self.fc(backward_first)
        decoder_boot = paddle.nn.functional.relu(decoder_boot)
        label_in = paddle.reshape(label_in, [-1])
        trg_embedding = self.embedding(label_in)

        trg_embedding = paddle.reshape(
            trg_embedding,
            [-1, Config.max_length, trg_embedding.shape[1]],
        )

        prediction = self.gru_decoder_with_attention(
            trg_embedding, encoded_vector, encoded_proj, decoder_boot
        )

        return prediction


class TestDygraphOCRAttention(unittest.TestCase):
    def test_ocr_test(self):
        seed = 90
        epoch_num = 1
        if core.is_compiled_with_cuda():
            batch_num = 3
        else:
            batch_num = 2
        np.random.seed = seed
        image_np = np.random.randn(
            Config.batch_size,
            Config.DATA_SHAPE[0],
            Config.DATA_SHAPE[1],
            Config.DATA_SHAPE[2],
        ).astype('float32')
        label_in_np = np.arange(0, Config.max_length, dtype='int64').reshape(
            [1, Config.max_length]
        )
        for i in range(2, Config.batch_size + 1):
            label_in_np = np.vstack(
                (
                    label_in_np,
                    np.arange(
                        (i - 1) * Config.max_length,
                        i * Config.max_length,
                        dtype='int64',
                    ).reshape([1, Config.max_length]),
                )
            )

        label_out_np = np.arange(0, Config.max_length, dtype='int64').reshape(
            [1, Config.max_length]
        )
        for i in range(2, Config.batch_size + 1):
            label_out_np = np.vstack(
                (
                    label_out_np,
                    np.arange(
                        (i - 1) * Config.max_length,
                        i * Config.max_length,
                        dtype='int64',
                    ).reshape([1, Config.max_length]),
                )
            )

        def run_dygraph():
            base.set_flags({'FLAGS_sort_sum_gradient': True})
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)
            ocr_attention = OCRAttention()

            if Config.learning_rate_decay == "piecewise_decay":
                learning_rate = paddle.optimizer.lr.piecewise_decay(
                    [50000], [Config.LR, Config.LR * 0.01]
                )
            else:
                learning_rate = Config.LR
            optimizer = paddle.optimizer.SGD(
                learning_rate=0.001, parameters=ocr_attention.parameters()
            )
            dy_param_init_value = {}
            for param in ocr_attention.parameters():
                dy_param_init_value[param.name] = param.numpy()
            for epoch in range(epoch_num):
                for batch_id in range(batch_num):
                    label_in = to_variable(label_in_np)
                    label_out = to_variable(label_out_np)
                    label_out.stop_gradient = True
                    img = to_variable(image_np)
                    dy_prediction = ocr_attention(img, label_in)
                    label_out = paddle.reshape(label_out, [-1, 1])
                    dy_prediction = paddle.reshape(
                        dy_prediction, [label_out.shape[0], -1]
                    )
                    loss = paddle.nn.functional.cross_entropy(
                        input=dy_prediction,
                        label=label_out,
                        reduction='none',
                        use_softmax=False,
                    )
                    avg_loss = paddle.sum(loss)

                    dy_out = avg_loss.numpy()

                    if epoch == 0 and batch_id == 0:
                        for param in ocr_attention.parameters():
                            if param.name not in dy_param_init_value:
                                dy_param_init_value[param.name] = param.numpy()
                    avg_loss.backward()
                    dy_grad_value = {}
                    for param in ocr_attention.parameters():
                        if param.trainable:
                            np_array = np.array(
                                param._grad_ivar().value().get_tensor()
                            )
                            dy_grad_value[
                                param.name + core.grad_var_suffix()
                            ] = np_array

                    optimizer.minimize(avg_loss)
                    ocr_attention.clear_gradients()
                    dy_param_value = {}
                    for param in ocr_attention.parameters():
                        dy_param_value[param.name] = param.numpy()

            return dy_out, dy_param_init_value, dy_param_value

        with base.dygraph.guard():
            dy_out, dy_param_init_value, dy_param_value = run_dygraph()

        with base.dygraph.guard():
            (
                eager_out,
                eager_param_init_value,
                eager_param_value,
            ) = run_dygraph()

        with new_program_scope():
            paddle.seed(seed)
            paddle.framework.random._manual_program_seed(seed)
            exe = base.Executor(
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )
            ocr_attention = OCRAttention()

            if Config.learning_rate_decay == "piecewise_decay":
                learning_rate = paddle.optimizer.lr.piecewise_decay(
                    [50000], [Config.LR, Config.LR * 0.01]
                )
            else:
                learning_rate = Config.LR

            optimizer = paddle.optimizer.SGD(learning_rate=0.001)

            images = paddle.static.data(
                name='pixel', shape=[-1] + Config.DATA_SHAPE, dtype='float32'
            )
            images.desc.set_need_check_feed(False)
            static_label_in = paddle.static.data(
                name='label_in', shape=[-1, 1], dtype='int64', lod_level=0
            )
            static_label_in.desc.set_need_check_feed(False)
            static_label_out = paddle.static.data(
                name='label_out', shape=[-1, 1], dtype='int64', lod_level=0
            )
            static_label_out.desc.set_need_check_feed(False)

            static_label_out.stop_gradient = True
            static_label_out.trainable = False

            static_prediction = ocr_attention(images, static_label_in)

            static_prediction = paddle.reshape(
                static_prediction, shape=[-1, Config.num_classes + 2]
            )

            cost = paddle.nn.functional.cross_entropy(
                input=static_prediction,
                label=static_label_out,
                reduction='none',
                use_softmax=False,
            )
            static_avg_loss = paddle.sum(cost)
            # param_grad_list = base.backward.append_backward(static_avg_loss)
            optimizer.minimize(static_avg_loss)

            static_param_init_value = {}
            static_param_name_list = []
            static_grad_name_list = []
            for param in ocr_attention.parameters():
                static_param_name_list.append(param.name)
                if param.trainable:
                    static_grad_name_list.append(
                        param.name + core.grad_var_suffix()
                    )

            out = exe.run(
                base.default_startup_program(),
                fetch_list=static_param_name_list,
            )

            for i in range(len(static_param_name_list)):
                static_param_init_value[static_param_name_list[i]] = out[i]

            fetch_list = [static_avg_loss.name]
            fetch_list.extend(static_param_name_list)
            fetch_list.extend(static_grad_name_list)
            for epoch in range(epoch_num):
                for batch_id in range(batch_num):
                    static_label_in = label_in_np
                    static_label_out = label_out_np
                    static_label_out = static_label_out.reshape((-1, 1))
                    out = exe.run(
                        base.default_main_program(),
                        feed={
                            "pixel": image_np,
                            "label_in": static_label_in,
                            "label_out": static_label_out,
                        },
                        fetch_list=fetch_list,
                    )
                    static_param_value = {}
                    static_grad_value = {}
                    static_out = out[0]
                    for i in range(1, len(static_param_name_list) + 1):
                        static_param_value[static_param_name_list[i - 1]] = out[
                            i
                        ]
                    grad_start_pos = len(static_param_name_list) + 1
                    for i in range(
                        grad_start_pos,
                        len(static_grad_name_list) + grad_start_pos,
                    ):
                        static_grad_value[
                            static_grad_name_list[i - grad_start_pos]
                        ] = out[i]

        np.testing.assert_allclose(static_out, dy_out, rtol=1e-05, atol=1e-8)

        for key, value in static_param_init_value.items():
            np.testing.assert_array_equal(value, dy_param_init_value[key])

        for key, value in static_param_value.items():
            np.testing.assert_allclose(
                value, dy_param_value[key], rtol=1e-05, atol=1e-8
            )

        # check eager here
        np.testing.assert_allclose(static_out, eager_out, rtol=1e-05, atol=1e-8)

        for key, value in static_param_init_value.items():
            np.testing.assert_array_equal(value, eager_param_init_value[key])

        for key, value in static_param_value.items():
            np.testing.assert_allclose(
                value, eager_param_value[key], rtol=1e-05, atol=1e-8
            )


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
