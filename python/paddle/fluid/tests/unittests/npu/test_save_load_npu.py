#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import sys
sys.path.append("..")
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.nn import Embedding
import paddle.fluid.framework as framework
from paddle.fluid.optimizer import Adam
from paddle.fluid.dygraph.base import to_variable
from test_imperative_base import new_program_scope
from paddle.fluid.executor import global_scope
import numpy as np
import six
import pickle
import os
import errno
from test_static_save_load import *

paddle.enable_static()


class PtbModel(fluid.Layer):
    def __init__(self,
                 name_scope,
                 hidden_size,
                 vocab_size,
                 num_layers=2,
                 num_steps=20,
                 init_scale=0.1,
                 dropout=None):
        super(PtbModel, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.init_scale = init_scale
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.dropout = dropout
        self.simple_lstm_rnn = SimpleLSTMRNN(
            self.full_name(),
            hidden_size,
            num_steps,
            num_layers=num_layers,
            init_scale=init_scale,
            dropout=dropout)
        self.embedding = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            weight_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale)))
        self.softmax_weight = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.hidden_size, self.vocab_size],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))
        self.softmax_bias = self.create_parameter(
            attr=fluid.ParamAttr(),
            shape=[self.vocab_size],
            dtype="float32",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale, high=self.init_scale))

    def forward(self, input, label, init_hidden, init_cell):
        init_h = fluid.layers.reshape(
            init_hidden, shape=[self.num_layers, -1, self.hidden_size])

        init_c = fluid.layers.reshape(
            init_cell, shape=[self.num_layers, -1, self.hidden_size])

        input = fluid.layers.cast(input, "int32")
        x_emb = self.embedding(input)
        x_emb = fluid.layers.reshape(
            x_emb, shape=[-1, self.num_steps, self.hidden_size])
        if self.dropout is not None and self.dropout > 0.0:
            x_emb = fluid.layers.dropout(
                x_emb,
                dropout_prob=self.drop_out,
                dropout_implementation='upscale_in_train')
        rnn_out, last_hidden, last_cell = self.simple_lstm_rnn(x_emb, init_h,
                                                               init_c)

        rnn_out = fluid.layers.reshape(
            rnn_out, shape=[-1, self.num_steps, self.hidden_size])
        projection = fluid.layers.matmul(rnn_out, self.softmax_weight)
        projection = fluid.layers.elementwise_add(projection, self.softmax_bias)
        projection = fluid.layers.reshape(
            projection, shape=[-1, self.vocab_size])
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=projection, label=label, soft_label=False)
        loss = fluid.layers.reshape(loss, shape=[-1, self.num_steps])
        loss = fluid.layers.reduce_mean(loss, dim=[0])
        loss = fluid.layers.reduce_sum(loss)

        return loss, last_hidden, last_cell


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUSaveLoadBase(TestSaveLoadBase):
    def set_place(self):
        return fluid.CPUPlace() if not core.is_compiled_with_npu(
        ) else paddle.NPUPlace(0)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUSaveLoadPartial(TestSaveLoadPartial):
    def set_place(self):
        return fluid.CPUPlace() if not core.is_compiled_with_npu(
        ) else paddle.NPUPlace(0)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUSaveLoadSetStateDict(TestSaveLoadSetStateDict):
    def set_place(self):
        return fluid.CPUPlace() if not core.is_compiled_with_npu(
        ) else paddle.NPUPlace(0)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUProgramStatePartial(TestProgramStatePartial):
    def set_place(self):
        return fluid.CPUPlace() if not core.is_compiled_with_npu(
        ) else paddle.NPUPlace(0)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUVariableInit(TestVariableInit):
    def set_place(self):
        return fluid.CPUPlace() if not core.is_compiled_with_npu(
        ) else paddle.NPUPlace(0)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPULoadFromOldInterface(TestLoadFromOldInterface):
    def set_place(self):
        return fluid.CPUPlace() if not core.is_compiled_with_npu(
        ) else paddle.NPUPlace(0)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPULoadFromOldInterfaceSingleFile(TestLoadFromOldInterfaceSingleFile):
    def set_place(self):
        return fluid.CPUPlace() if not core.is_compiled_with_npu(
        ) else paddle.NPUPlace(0)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUProgramStateOldSave(TestProgramStateOldSave):
    def set_place(self):
        return fluid.CPUPlace() if not core.is_compiled_with_npu(
        ) else paddle.NPUPlace(0)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNPUProgramStateOldSaveSingleModel(TestProgramStateOldSaveSingleModel):
    def set_place(self):
        return fluid.CPUPlace() if not core.is_compiled_with_npu(
        ) else paddle.NPUPlace(0)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
