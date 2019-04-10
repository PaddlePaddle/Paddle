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

decoder_size = 128

word_vector_dim = 128
max_length = 2
sos = 0
eos = 1
gradient_clip = 10
LR = 1.0
learning_rate_decay = None
batch_size = 32

num_classes = 95

SOS = 0
EOS = 1

DATA_DIR_NAME = "/paddle/slf/dataset/ctc_data/data"
TRAIN_DATA_DIR_NAME = "train_images"
TRAIN_LIST_FILE_NAME = "train.list"
DATA_SHAPE = [1, 48, 384]


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
        return context, decoder_state_fc


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
        #self.fc_test = FC(self.full_name(), size=decoder_size,
        #        param_attr=None, bias_attr=None)
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
        # if framework._in_dygraph_mode():
        for i in range(target_embedding.shape[1]):
            current_word = fluid.layers.slice(
                target_embedding, axes=[1], starts=[i], ends=[i + 1])
            current_word = fluid.layers.reshape(
                current_word, [-1, current_word.shape[2]], inplace=False)

            context, test1 = self.simple_attention(encoder_vec, encoder_proj,
                                                   hidden_mem)
            fc_1 = self.fc_1_layer(context)
            fc_2 = self.fc_2_layer(current_word)
            decoder_inputs = fluid.layers.elementwise_add(x=fc_1, y=fc_2)

            h, _, _ = self.gru_unit(decoder_inputs, hidden_mem)
            hidden_mem = h
            if i == 0:
                test = h

            out = self.out_layer(h)
            res.append(out)

        res1 = fluid.layers.concat(res, axis=0)

        return res1, test


class MLP(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(MLP, self).__init__(name_scope)
        self._fc1 = FC(self.full_name(),
                       3,
                       param_attr=fluid.ParamAttr(
                           initializer=fluid.initializer.Constant(value=0.1)),
                       bias_attr=fluid.ParamAttr(
                           initializer=fluid.initializer.Constant(value=0.1)))
        self._fc2 = FC(self.full_name(),
                       4,
                       param_attr=fluid.ParamAttr(
                           initializer=fluid.initializer.Constant(value=0.1)),
                       bias_attr=fluid.ParamAttr(
                           initializer=fluid.initializer.Constant(value=0.1)))

    def forward(self, inputs):
        x = self._fc1(inputs)
        x = self._fc2(x)
        x = fluid.layers.reduce_sum(x)
        return x


class TestDygraphOCRAttention(unittest.TestCase):
    def test_while_op(self):
        seed = 90
        epoch_num = 1
        batch_num = 1
        np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        with fluid.dygraph.guard():
            var_inp = fluid.dygraph.base.to_variable(np_inp)
            mlp = MLP("mlp")
            out = mlp(var_inp)
            dy_out = out._numpy()
            out._backward()
            dy_grad = mlp._fc1._w._gradient()

        with new_program_scope():
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            mlp = MLP("mlp")
            out = mlp(inp)
            param_grads = fluid.backward.append_backward(
                out, parameter_list=[mlp._fc1._w.name])[0]
            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))
            exe.run(fluid.default_startup_program())

            static_out, static_grad = exe.run(
                feed={inp.name: np_inp},
                fetch_list=[out.name, param_grads[1].name])

        self.assertTrue(np.allclose(dy_out, static_out))
        self.assertTrue(np.allclose(dy_grad, static_grad))

        # trg_embedding_np = np.random.randn(32, max_length,
        #                                    decoder_size).astype('float32')
        # encoder_vec_np = np.random.randn(32, 48, 400).astype('float32')
        # encoder_proj_np = np.random.randn(32, 48,
        #                                   decoder_size).astype('float32')
        # decoder_boot_np = np.random.randn(32, decoder_size).astype('float32')
        # label_out_np = np.random.randn(32 * max_length, 1).astype('int64')
        #
        # with fluid.dygraph.guard(fluid.CPUPlace()):
        #     fluid.default_startup_program().random_seed = seed
        #     fluid.default_main_program().random_seed = seed
        #     ocr_attention = GRUDecoderWithAttention("ocr_attention",
        #                                             decoder_size, 95)
        #     if learning_rate_decay == "piecewise_decay":
        #         learning_rate = fluid.layers.piecewise_decay([50000],
        #                                                      [LR, LR * 0.01])
        #     else:
        #         learning_rate = LR
        #     optimizer = fluid.optimizer.Adadelta(
        #         learning_rate=learning_rate, epsilon=1.0e-6, rho=0.9)
        #     # optimizer = fluid.optimizer.SGD(learning_rate=0.01)
        #     place = fluid.CPUPlace()
        #     dy_param_init_value = {}
        #     for param in ocr_attention.parameters():
        #         dy_param_init_value[param.name] = param._numpy()
        #     label_out = to_variable(label_out_np)
        #
        #     label_out._stop_gradient = True
        #     label_out.trainable = False
        #     trg_embedding = to_variable(trg_embedding_np)
        #     encoder_vec = to_variable(encoder_vec_np)
        #     encoder_proj = to_variable(encoder_proj_np)
        #     decoder_boot = to_variable(decoder_boot_np)
        #     for i in range(batch_num):
        #         dy_prediction, dy_grad = ocr_attention(
        #             trg_embedding, encoder_vec, encoder_proj, decoder_boot)
        #         loss = fluid.layers.cross_entropy(
        #             input=dy_prediction, label=label_out)
        #         avg_loss = fluid.layers.reduce_sum(loss)
        #
        #         dy_out = dy_prediction._numpy()
        #         if i == 0:
        #             for param in ocr_attention.parameters():
        #                 if param.name not in dy_param_init_value:
        #                     dy_param_init_value[param.name] = param._numpy()
        #
        #         avg_loss._backward()
        #         dy_grad_test = dy_grad._ivar._grad_name()
        #
        #         dy_grad_value = {}
        #         for param in ocr_attention.parameters():
        #             if param.trainable:
        #                 np_array = np.array(param._ivar._grad_ivar().value()
        #                                     .get_tensor())
        #                 dy_grad_value[param.name + core.grad_var_suffix(
        #                 )] = np_array
        #
        #         #optimizer.minimize(avg_loss)
        #         ocr_attention.clear_gradients()
        #         dy_param_value = {}
        #         for param in ocr_attention.parameters():
        #             dy_param_value[param.name] = param._numpy()
        #
        # with new_program_scope():
        #     fluid.default_startup_program().random_seed = seed
        #     fluid.default_main_program().random_seed = seed
        #     print("static start")
        #
        #     exe = fluid.Executor(fluid.CPUPlace())
        #     #        if not core.is_compiled_with_cuda() else fluid.CUDAPlace(1))
        #     ocr_attention = GRUDecoderWithAttention("ocr_attention",
        #                                             decoder_size, 95)
        #
        #     if learning_rate_decay == "piecewise_decay":
        #         learning_rate = fluid.layers.piecewise_decay([50000],
        #                                                      [LR, LR * 0.01])
        #     else:
        #         learning_rate = LR
        #     optimizer = fluid.optimizer.Adadelta(
        #         learning_rate=learning_rate, epsilon=1.0e-6, rho=0.9)
        #     # optimizer = fluid.optimizer.SGD(learning_rate=0.01)
        #     place = fluid.CPUPlace()
        #
        #     trg_embedding = fluid.layers.data(
        #         name='trg_embedding',
        #         shape=[-1, max_length, decoder_size],
        #         dtype='float32')
        #     encoder_vec = fluid.layers.data(
        #         name='encoder_vec', shape=[-1, 48, 400], dtype='float32')
        #     encoder_proj = fluid.layers.data(
        #         name='encoder_proj',
        #         shape=[-1, 48, decoder_size],
        #         dtype='float32')
        #     decoder_boot = fluid.layers.data(
        #         name='decoder_boot', shape=[-1, decoder_size], dtype='float32')
        #     static_label_out = fluid.layers.data(
        #         name='label_out', shape=[-1, 1], dtype='int64')
        #
        #     static_label_out._stop_gradient = True
        #     static_label_out.trainable = False
        #
        #     static_prediction, static_out = ocr_attention(
        #         trg_embedding, encoder_vec, encoder_proj, decoder_boot)
        #
        #     cost = fluid.layers.cross_entropy(
        #         input=static_prediction, label=static_label_out)
        #     static_avg_loss = fluid.layers.reduce_sum(cost)
        #     param_grad_list = fluid.backward.append_backward(static_avg_loss)
        #
        #     # print(fluid.framework.default_main_program().block(0).ops)
        #     # optimizer.minimize(static_avg_loss)
        #
        #     static_param_init_value = {}
        #     static_param_name_list = []
        #     static_grad_name_list = []
        #     for param in ocr_attention.parameters():
        #         static_param_name_list.append(param.name)
        #         if param.trainable:
        #             static_grad_name_list.append(param.name +
        #                                          core.grad_var_suffix())
        #
        #     out = exe.run(fluid.default_startup_program(),
        #                   fetch_list=static_param_name_list)
        #
        #     for i in range(len(static_param_name_list)):
        #         static_param_init_value[static_param_name_list[i]] = out[i]
        #
        #     for i in range(batch_num):
        #         fetch_list = [static_prediction.name, static_out.name + '@GRAD']
        #         fetch_list.extend(static_param_name_list)
        #         fetch_list.extend(static_grad_name_list)
        #         out = exe.run(fluid.default_main_program(),
        #                       feed={
        #                           "trg_embedding": trg_embedding_np,
        #                           "encoder_vec": encoder_vec_np,
        #                           "encoder_proj": encoder_proj_np,
        #                           "decoder_boot": decoder_boot_np,
        #                           "label_out": label_out_np
        #                       },
        #                       fetch_list=fetch_list)
        #         static_param_value = {}
        #         static_grad_value = {}
        #         static_out = out[0]
        #         static_grad_test = out[1]
        #         for i in range(2, len(static_param_name_list) + 1):
        #             static_param_value[static_param_name_list[i - 2]] = out[i]
        #         grad_start_pos = len(static_param_name_list) + 2
        #         for i in range(grad_start_pos,
        #                        len(static_grad_name_list) + grad_start_pos):
        #             static_grad_value[static_grad_name_list[
        #                 i - grad_start_pos]] = out[i]
        #
        # for key, value in six.iteritems(static_param_init_value):
        #     self.assertTrue(np.allclose(value, dy_param_init_value[key]))
        # '''
        # for i in range(len(static_loss_grad)):
        #     #for j in range(len(static_loss_grad[0])):
        #     if not np.allclose(static_loss_grad[i], avg_loss_grad[i]):
        #         print("{} {} {}".format(i, static_loss_grad[i][:20], avg_loss_grad[i][:20]))
        # '''
        # print("+++++++++++++loss+++++++++++++++")
        # if not np.array_equal(static_out, dy_out):
        #     print("not ok")
        # else:
        #     print(static_out)
        #     print("ok")
        #
        # if not np.array_equal(static_grad_test, dy_grad_test):
        #     print(static_grad_test)
        #     print(dy_grad_test)
        #     print("not ok")
        # else:
        #     print("ok")
        #
        # print("++++++++++++parameter+++++++++++")
        #
        # for key, value in six.iteritems(static_param_value):
        #     if not np.array_equal(value, dy_param_value[key]):
        #         print("{} is not ok".format(key))
        #         pass
        #     else:
        #         #print("{} is ok".format(key))
        #         pass
        #     # self.assertTrue(np.allclose(value, dy_param_value[key], atol=1e-5))
        #
        # print("++++++++++++++gradient+++++++++++")
        # for key, value in six.iteritems(static_grad_value):
        #     if not np.array_equal(value, dy_grad_value[key]):
        #         #for i in range(len(value)):
        #         #    if not np.array_equal(value[i], dy_grad_value[key][i]):
        #         # print("{} {}".format(value[0], dy_grad_value[key][0]))
        #         print('{} gradient is not ok'.format(key))
        #         pass
        #     else:
        #         # print("{} gradient is ok".format(key))
        #         pass


if __name__ == '__main__':
    unittest.main()
