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
from __future__ import print_function

import contextlib
import unittest
import numpy as np
import six
import os
from PIL import Image
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC, BatchNorm, Embedding, GRUUnit
from paddle.fluid.dygraph.base import to_variable
from test_imperative_base import new_program_scope


class Config(object):
    '''
    config for training
    '''
    # decoder size for decoder stage
    decoder_size = 128
    # size for word embedding
    word_vector_dim = 128
    # max length for label padding
    max_length = 15
    # optimizer setting
    LR = 1.0
    learning_rate_decay = None

    # batch size to train
    batch_size = 32
    # class number to classify
    num_classes = 95
    # special label for start and end
    SOS = 0
    EOS = 1
    # settings for ctc data, not use in unittest
    DATA_DIR_NAME = "./dataset/ctc_data/data"
    TRAIN_DATA_DIR_NAME = "train_images"
    TRAIN_LIST_FILE_NAME = "train.list"

    # data shape for input image
    DATA_SHAPE = [1, 48, 384]


class DataReader(object):
    '''
    train image reader for ctc data, not use in unittest
    '''

    def train_reader(self,
                     img_root_dir,
                     img_label_list,
                     batchsize,
                     cycle,
                     shuffle=False):
        '''
        read data from img_root_dir and img_label_list
        '''
        img_label_lines = []
        to_file = "tmp.txt"
        if not shuffle:
            cmd = "cat " + img_label_list + " | awk '{print $1,$2,$3,$4;}' > " + to_file
        elif batchsize == 1:
            cmd = "cat " + img_label_list + " | awk '{print $1,$2,$3,$4;}' | shuf > " + to_file
        else:
            cmd = "cat " + img_label_list + " | awk '{printf(\"%04d%.4f %s\\n\", $1, rand(), $0)}' | sort | sed 1,$((1 + RANDOM % 100))d | "
            cmd += "awk '{printf $2\" \"$3\" \"$4\" \"$5\" \"; if(NR % " + str(
                batchsize) + " == 0) print \"\";}' | shuf | "
            cmd += "awk '{if(NF == " + str(
                batchsize
            ) + " * 4) {for(i=0; i < " + str(
                batchsize
            ) + "; i++) print $(4*i+1)\" \"$(4*i+2)\" \"$(4*i+3)\" \"$(4*i+4);}}' > " + to_file
        os.system(cmd)
        img_label_lines = open(to_file, 'r').readlines()

        def reader():
            sizes = len(img_label_lines) // batchsize
            if sizes == 0:
                raise ValueError('batchsize is bigger than the dataset size.')
            while True:
                for i in range(sizes):
                    result = []
                    sz = [0, 0]
                    max_len = 0
                    for k in range(batchsize):
                        line = img_label_lines[i * batchsize + k]
                        items = line.split(' ')
                        label = [int(c) for c in items[-1].split(',')]
                        max_len = max(max_len, len(label))

                    for j in range(batchsize):
                        line = img_label_lines[i * batchsize + j]
                        items = line.split(' ')
                        label = [int(c) for c in items[-1].split(',')]
                        if max_length > len(label) + 1:
                            extend_label = [EOS] * (max_length - len(label) - 1)
                            label.extend(extend_label)
                        else:
                            label = label[0:max_length - 1]
                        img = Image.open(os.path.join(img_root_dir, items[
                            2])).convert('L')
                        if j == 0:
                            sz = img.size
                        img = img.resize((sz[0], sz[1]))
                        img = np.array(img) - 127.5
                        img = img[np.newaxis, ...]
                        result.append([img, [SOS] + label, label + [EOS]])
                    yield result
                if not cycle:
                    break

        return reader

    def get_attention_data(self, data, place, need_label=True):
        pixel_tensor = core.LoDTensor()
        pixel_data = None
        pixel_data = np.concatenate(
            list(map(lambda x: x[0][np.newaxis, :], data)),
            axis=0).astype("float32")
        pixel_tensor.set(pixel_data, place)
        label_in_tensor = core.LoDTensor()
        label_out_tensor = core.LoDTensor()
        label_in = list(map(lambda x: x[1], data))
        label_out = list(map(lambda x: x[2], data))
        label_in_tensor.set(label_in, place)
        label_out_tensor.set(label_out, place)
        return pixel_tensor, label_in_tensor, label_out_tensor


class ConvBNPool(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 group,
                 out_ch,
                 channels,
                 act="relu",
                 is_test=False,
                 pool=True,
                 use_cudnn=True):
        super(ConvBNPool, self).__init__(name_scope)
        self.group = group
        self.pool = pool

        filter_size = 3
        conv_std_0 = (2.0 / (filter_size**2 * channels[0]))**0.5
        conv_param_0 = fluid.ParamAttr(
            initializer=fluid.initializer.Normal(0.0, conv_std_0))

        conv_std_1 = (2.0 / (filter_size**2 * channels[1]))**0.5
        conv_param_1 = fluid.ParamAttr(
            initializer=fluid.initializer.Normal(0.0, conv_std_1))
        bias_attr_0 = fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.0))
        bias_attr_1 = fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.0))

        self.conv_0_layer = Conv2D(
            self.full_name(),
            channels[0],
            out_ch[0],
            3,
            padding=1,
            param_attr=conv_param_0,
            bias_attr=bias_attr_0,
            act=None,
            use_cudnn=use_cudnn)
        self.bn_0_layer = BatchNorm(
            self.full_name(), out_ch[0], act=act, is_test=is_test)
        self.conv_1_layer = Conv2D(
            self.full_name(),
            num_channels=channels[1],
            num_filters=out_ch[1],
            filter_size=3,
            padding=1,
            param_attr=conv_param_1,
            bias_attr=bias_attr_1,
            act=None,
            use_cudnn=use_cudnn)
        self.bn_1_layer = BatchNorm(
            self.full_name(), out_ch[1], act=act, is_test=is_test)

        if self.pool:
            self.pool_layer = Pool2D(
                self.full_name(),
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
    def __init__(self, name_scope, is_test=False, use_cudnn=True):
        super(OCRConv, self).__init__(name_scope)
        self.conv_bn_pool_1 = ConvBNPool(
            self.full_name(),
            2, [16, 16], [1, 16],
            is_test=is_test,
            use_cudnn=use_cudnn)
        self.conv_bn_pool_2 = ConvBNPool(
            self.full_name(),
            2, [32, 32], [16, 32],
            is_test=is_test,
            use_cudnn=use_cudnn)
        self.conv_bn_pool_3 = ConvBNPool(
            self.full_name(),
            2, [64, 64], [32, 64],
            is_test=is_test,
            use_cudnn=use_cudnn)
        self.conv_bn_pool_4 = ConvBNPool(
            self.full_name(),
            2, [128, 128], [64, 128],
            is_test=is_test,
            pool=False,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        inputs_1 = self.conv_bn_pool_1(inputs)
        inputs_2 = self.conv_bn_pool_2(inputs_1)
        inputs_3 = self.conv_bn_pool_3(inputs_2)
        inputs_4 = self.conv_bn_pool_4(inputs_3)

        return inputs_4


class DynamicGRU(fluid.dygraph.Layer):
    def __init__(self,
                 scope_name,
                 size,
                 param_attr=None,
                 bias_attr=None,
                 is_reverse=False,
                 gate_activation='sigmoid',
                 candidate_activation='tanh',
                 h_0=None,
                 origin_mode=False):
        super(DynamicGRU, self).__init__(scope_name)

        self.gru_unit = GRUUnit(
            self.full_name(),
            size * 3,
            param_attr=param_attr,
            bias_attr=bias_attr,
            activation=candidate_activation,
            gate_activation=gate_activation,
            origin_mode=origin_mode)

        self.h_0 = h_0
        self.is_reverse = is_reverse

    def forward(self, inputs):
        hidden = self.h_0
        res = []
        for i in range(inputs.shape[1]):
            if self.is_reverse:
                i = inputs.shape[1] - 1 - i
            input_ = fluid.layers.slice(
                inputs, axes=[1], starts=[i], ends=[i + 1])
            input_ = fluid.layers.reshape(input_, [-1, input_.shape[2]])
            hidden, reset, gate = self.gru_unit(input_, hidden)
            hidden_ = fluid.layers.reshape(hidden, [-1, 1, hidden.shape[1]])
            if self.is_reverse:
                res = [hidden_] + res
            else:
                res.append(hidden_)
        res = fluid.layers.concat(res, axis=1)
        return res


class EncoderNet(fluid.dygraph.Layer):
    def __init__(self,
                 scope_name,
                 rnn_hidden_size=200,
                 is_test=False,
                 use_cudnn=True):
        super(EncoderNet, self).__init__(scope_name)
        self.rnn_hidden_size = rnn_hidden_size
        para_attr = fluid.ParamAttr(initializer=fluid.initializer.Normal(0.0,
                                                                         0.02))
        bias_attr = fluid.ParamAttr(
            initializer=fluid.initializer.Normal(0.0, 0.02), learning_rate=2.0)
        if fluid.framework._in_dygraph_mode():
            h_0 = np.zeros(
                (Config.batch_size, rnn_hidden_size), dtype="float32")
            h_0 = to_variable(h_0)
        else:
            h_0 = fluid.layers.fill_constant(
                shape=[Config.batch_size, rnn_hidden_size],
                dtype='float32',
                value=0)
        self.ocr_convs = OCRConv(
            self.full_name(), is_test=is_test, use_cudnn=use_cudnn)

        self.fc_1_layer = FC(self.full_name(),
                             rnn_hidden_size * 3,
                             param_attr=None,
                             bias_attr=False,
                             num_flatten_dims=2)
        self.fc_2_layer = FC(self.full_name(),
                             rnn_hidden_size * 3,
                             param_attr=para_attr,
                             bias_attr=False,
                             num_flatten_dims=2)
        self.gru_forward_layer = DynamicGRU(
            self.full_name(),
            size=rnn_hidden_size,
            h_0=h_0,
            param_attr=para_attr,
            bias_attr=bias_attr,
            candidate_activation='relu')
        self.gru_backward_layer = DynamicGRU(
            self.full_name(),
            size=rnn_hidden_size,
            h_0=h_0,
            param_attr=para_attr,
            bias_attr=bias_attr,
            candidate_activation='relu',
            is_reverse=True)

        self.encoded_proj_fc = FC(self.full_name(),
                                  Config.decoder_size,
                                  param_attr=para_attr,
                                  bias_attr=False,
                                  num_flatten_dims=2)

    def forward(self, inputs):
        conv_features = self.ocr_convs(inputs)
        #sliced_feature = fluid.layers.im2sequence(
        #    input=conv_features,
        #    stride=[1, 1],
        #    filter_size=[conv_features.shape[2], 1])

        sliced_feature = fluid.layers.reshape(
            conv_features,
            [-1, 48, conv_features.shape[1] * conv_features.shape[2]],
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
    def __init__(self, scope_name, decoder_size):
        super(SimpleAttention, self).__init__(scope_name)
        para_attr = fluid.ParamAttr(initializer=fluid.initializer.Normal(0.0,
                                                                         0.02))
        bias_attr = fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.0))

        self.fc_1 = FC(self.full_name(),
                       decoder_size,
                       act=None,
                       param_attr=para_attr,
                       bias_attr=bias_attr)
        self.fc_2 = FC(self.full_name(),
                       1,
                       act=None,
                       param_attr=para_attr,
                       bias_attr=bias_attr)

    def _build_once(self, encoder_vec, encoder_proj, decoder_state):
        pass

    def forward(self, encoder_vec, encoder_proj, decoder_state):

        decoder_state_fc = self.fc_1(decoder_state)
        decoder_state_proj_reshape = fluid.layers.reshape(
            decoder_state_fc, [-1, 1, decoder_state_fc.shape[1]], inplace=False)
        decoder_state_expand = fluid.layers.expand(
            decoder_state_proj_reshape, [1, encoder_proj.shape[1], 1])
        concated = fluid.layers.elementwise_add(encoder_proj,
                                                decoder_state_expand)
        concated = fluid.layers.tanh(x=concated)
        attention_weight = self.fc_2(concated)
        weights_reshape = fluid.layers.reshape(
            x=attention_weight, shape=[-1], inplace=False)
        scaled = fluid.layers.elementwise_mul(
            x=encoder_vec, y=weights_reshape, axis=0)
        scaled = fluid.layers.transpose(scaled, [0, 2, 1])
        scaled = fluid.layers.reshape(
            scaled, [-1, scaled.shape[1], scaled.shape[2], 1])
        context = fluid.layers.pool2d(
            input=scaled,
            pool_size=[scaled.shape[2], scaled.shape[3]],
            pool_type='avg')
        context = fluid.layers.reshape(
            context, [-1, context.shape[1]], inplace=False)
        return context


class GRUDecoderWithAttention(fluid.dygraph.Layer):
    def __init__(self, scope_name, decoder_size, num_classes):
        super(GRUDecoderWithAttention, self).__init__(scope_name)
        self.simple_attention = SimpleAttention(self.full_name(), decoder_size)
        para_attr = fluid.ParamAttr(initializer=fluid.initializer.Normal(0.0,
                                                                         0.02))

        bias_attr = fluid.ParamAttr(
            initializer=fluid.initializer.Constant(value=0.0))

        self.fc_1_layer = FC(self.full_name(),
                             size=decoder_size * 3,
                             param_attr=para_attr,
                             bias_attr=None)
        self.fc_2_layer = FC(self.full_name(),
                             size=decoder_size * 3,
                             param_attr=para_attr,
                             bias_attr=None)
        self.gru_unit = GRUUnit(
            self.full_name(),
            size=decoder_size * 3,
            param_attr=para_attr,
            bias_attr=bias_attr)
        self.out_layer = FC(self.full_name(),
                            size=num_classes + 2,
                            param_attr=None,
                            bias_attr=None,
                            act='softmax')

        self.decoder_size = decoder_size

    def _build_once(self, target_embedding, encoder_vec, encoder_proj,
                    decoder_boot):
        pass

    def forward(self, target_embedding, encoder_vec, encoder_proj,
                decoder_boot):
        res = []
        hidden_mem = decoder_boot
        for i in range(target_embedding.shape[1]):
            current_word = fluid.layers.slice(
                target_embedding, axes=[1], starts=[i], ends=[i + 1])
            current_word = fluid.layers.reshape(
                current_word, [-1, current_word.shape[2]], inplace=False)

            context = self.simple_attention(encoder_vec, encoder_proj,
                                            hidden_mem)
            fc_1 = self.fc_1_layer(context)
            fc_2 = self.fc_2_layer(current_word)
            decoder_inputs = fluid.layers.elementwise_add(x=fc_1, y=fc_2)

            h, _, _ = self.gru_unit(decoder_inputs, hidden_mem)
            hidden_mem = h
            out = self.out_layer(h)
            res.append(out)

        res1 = fluid.layers.concat(res, axis=0)

        return res1


class OCRAttention(fluid.dygraph.Layer):
    def __init__(self, scope_name):
        super(OCRAttention, self).__init__(scope_name)
        self.encoder_net = EncoderNet(self.full_name())
        self.fc = FC(self.full_name(),
                     size=Config.decoder_size,
                     bias_attr=False,
                     act='relu')
        self.embedding = Embedding(
            self.full_name(), [Config.num_classes + 2, Config.word_vector_dim],
            dtype='float32')
        self.gru_decoder_with_attention = GRUDecoderWithAttention(
            self.full_name(), Config.decoder_size, Config.num_classes)

    def _build_once(self, inputs, label_in):
        pass

    def forward(self, inputs, label_in):
        gru_backward, encoded_vector, encoded_proj = self.encoder_net(inputs)
        backward_first = fluid.layers.slice(
            gru_backward, axes=[1], starts=[0], ends=[1])
        backward_first = fluid.layers.reshape(
            backward_first, [-1, backward_first.shape[2]], inplace=False)
        decoder_boot = self.fc(backward_first)
        label_in = fluid.layers.reshape(label_in, [-1, 1])
        trg_embedding = self.embedding(label_in)
        trg_embedding = fluid.layers.reshape(
            trg_embedding, [-1, Config.max_length, trg_embedding.shape[1]],
            inplace=False)
        prediction = self.gru_decoder_with_attention(
            trg_embedding, encoded_vector, encoded_proj, decoder_boot)

        return prediction


class TestDygraphOCRAttention(unittest.TestCase):
    def test_while_op(self):
        seed = 90
        epoch_num = 1
        batch_num = 1
        np.random.seed = seed
        image_np = np.random.randn(Config.batch_size, Config.DATA_SHAPE[0],
                                   Config.DATA_SHAPE[1],
                                   Config.DATA_SHAPE[2]).astype('float32')
        label_in_np = np.random.randint(
            0,
            Config.num_classes + 1,
            size=(Config.batch_size, Config.max_length),
            dtype='int64')
        label_out_np = np.random.randint(
            0,
            Config.num_classes + 1,
            size=(Config.batch_size, Config.max_length),
            dtype='int64')

        with fluid.dygraph.guard(fluid.CPUPlace()):
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            ocr_attention = OCRAttention("ocr_attention")

            if Config.learning_rate_decay == "piecewise_decay":
                learning_rate = fluid.layers.piecewise_decay(
                    [50000], [Config.LR, Config.LR * 0.01])
            else:
                learning_rate = Config.LR
            #optimizer = fluid.optimizer.Adadelta(learning_rate=learning_rate,
            #    epsilon=1.0e-6, rho=0.9)
            optimizer = fluid.optimizer.SGD(learning_rate=0.001)
            place = fluid.CPUPlace()
            dy_param_init_value = {}
            for param in ocr_attention.parameters():
                dy_param_init_value[param.name] = param._numpy()
            for epoch in range(epoch_num):
                for batch_id in range(batch_num):
                    label_in = to_variable(label_in_np)
                    label_out = to_variable(label_out_np)

                    label_out._stop_gradient = True
                    label_out.trainable = False
                    img = to_variable(image_np)
                    dy_prediction = ocr_attention(img, label_in)
                    label_out = fluid.layers.reshape(label_out, [-1, 1])
                    loss = fluid.layers.cross_entropy(
                        input=dy_prediction, label=label_out)
                    avg_loss = fluid.layers.reduce_sum(loss)

                    dy_out = avg_loss._numpy()

                    if epoch == 0 and batch_id == 0:
                        for param in ocr_attention.parameters():
                            if param.name not in dy_param_init_value:
                                dy_param_init_value[param.name] = param._numpy()
                    avg_loss._backward()

                    dy_grad_value = {}
                    for param in ocr_attention.parameters():
                        if param.trainable:
                            np_array = np.array(param._ivar._grad_ivar().value()
                                                .get_tensor())
                            dy_grad_value[param.name + core.grad_var_suffix(
                            )] = np_array

                    optimizer.minimize(avg_loss)
                    ocr_attention.clear_gradients()
                    dy_param_value = {}
                    for param in ocr_attention.parameters():
                        dy_param_value[param.name] = param._numpy()

        with new_program_scope():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            # print("static start")
            exe = fluid.Executor(fluid.CPUPlace())
            ocr_attention = OCRAttention("ocr_attention")

            if Config.learning_rate_decay == "piecewise_decay":
                learning_rate = fluid.layers.piecewise_decay(
                    [50000], [Config.LR, Config.LR * 0.01])
            else:
                learning_rate = Config.LR
            #optimizer = fluid.optimizer.Adadelta(learning_rate=learning_rate,
            #    epsilon=1.0e-6, rho=0.9)
            optimizer = fluid.optimizer.SGD(learning_rate=0.001)

            place = fluid.CPUPlace()

            images = fluid.layers.data(
                name='pixel', shape=Config.DATA_SHAPE, dtype='float32')
            static_label_in = fluid.layers.data(
                name='label_in', shape=[1], dtype='int64', lod_level=0)
            static_label_out = fluid.layers.data(
                name='label_out', shape=[1], dtype='int64', lod_level=0)
            static_label_out._stop_gradient = True
            static_label_out.trainable = False

            static_prediction = ocr_attention(images, static_label_in)

            cost = fluid.layers.cross_entropy(
                input=static_prediction, label=static_label_out)
            static_avg_loss = fluid.layers.reduce_sum(cost)
            # param_grad_list = fluid.backward.append_backward(static_avg_loss)
            optimizer.minimize(static_avg_loss)

            static_param_init_value = {}
            static_param_name_list = []
            static_grad_name_list = []
            for param in ocr_attention.parameters():
                static_param_name_list.append(param.name)
                if param.trainable:
                    static_grad_name_list.append(param.name +
                                                 core.grad_var_suffix())

            out = exe.run(fluid.default_startup_program(),
                          fetch_list=static_param_name_list)

            for i in range(len(static_param_name_list)):
                static_param_init_value[static_param_name_list[i]] = out[i]

            for epoch in range(epoch_num):
                for batch_id in range(batch_num):
                    fetch_list = [static_avg_loss.name]
                    fetch_list.extend(static_param_name_list)
                    fetch_list.extend(static_grad_name_list)
                    static_label_in = label_in_np
                    static_label_out = label_out_np
                    static_label_out = static_label_out.reshape((-1, 1))
                    out = exe.run(fluid.default_main_program(),
                                  feed={
                                      "pixel": image_np,
                                      "label_in": static_label_in,
                                      "label_out": static_label_out
                                  },
                                  fetch_list=fetch_list)
                    static_param_value = {}
                    static_grad_value = {}
                    static_out = out[0]
                    for i in range(1, len(static_param_name_list) + 1):
                        static_param_value[static_param_name_list[i - 1]] = out[
                            i]
                    grad_start_pos = len(static_param_name_list) + 1
                    for i in range(grad_start_pos,
                                   len(static_grad_name_list) + grad_start_pos):
                        static_grad_value[static_grad_name_list[
                            i - grad_start_pos]] = out[i]

        for key, value in six.iteritems(static_param_init_value):
            self.assertTrue(np.allclose(value, dy_param_init_value[key]))
        self.assertTrue(np.allclose(static_out, dy_out, atol=1e-5))

        for key, value in six.iteritems(static_param_value):
            self.assertTrue(np.allclose(value, dy_param_value[key], atol=1e-5))

        for key, value in six.iteritems(static_grad_value):
            self.assertTrue(np.allclose(value, dy_grad_value[key], atol=1e-5))


if __name__ == '__main__':
    unittest.main()
