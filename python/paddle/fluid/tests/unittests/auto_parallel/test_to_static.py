# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.distributed.auto_parallel as auto
import paddle.distributed.fleet as fleet

from paddle import LazyGuard
from paddle.io import Dataset
from paddle.static import InputSpec
from paddle.fluid.framework import _non_static_mode
from paddle.distributed.auto_parallel.engine import Engine
from paddle.distributed.auto_parallel.helper import ProgramHelper

batch_size = 4
batch_num = 30
hidden_size = 1024
class_num = 10


class MyDataset(Dataset):

    def __init__(self, num_samples):
        super(MyDataset, self).__init__()
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=hidden_size).astype("float32")
        label = np.random.randint(0, class_num - 1, dtype="int64")
        return input, label

    def __len__(self):
        return self.num_samples


class MLPLayer(nn.Layer):

    def __init__(self,
                 hidden_size=1024,
                 intermediate_size=4 * 1024,
                 dropout_ratio=0.1,
                 initializer_range=0.02):
        super(MLPLayer, self).__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=initializer_range))

        self.linear0 = nn.Linear(d_model,
                                 dim_feedforward,
                                 weight_attr,
                                 bias_attr=None)
        self.linear1 = nn.Linear(dim_feedforward,
                                 d_model,
                                 weight_attr,
                                 bias_attr=None)
        self.linear2 = nn.Linear(d_model, 1, weight_attr, bias_attr=None)
        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)
        self.dropout = nn.Dropout(dropout_ratio, mode="upscale_in_train")

    def forward(self, input):
        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        out = self.dropout(out)
        out = self.linear2(out)

        return out


class TestWholeProgram(unittest.TestCase):

    def test_apply_optimzier(self):
        paddle.disable_static()
        mlp = MLPLayer(hidden_size=hidden_size,
                       intermediate_size=4 * hidden_size,
                       dropout_ratio=0.1,
                       initializer_range=0.02)
        metrics = paddle.metric.Accuracy()
        loss = paddle.nn.CrossEntropyLoss()
        optimizer = paddle.optimizer.SGD(learning_rate=0.00001,
                                         parameters=mlp.parameters())
        inputs = InputSpec([batch_size, hidden_size], 'float32', 'x')
        labels = InputSpec([batch_size], 'int64', 'label')

        program_helper = ProgramHelper(mlp, loss, [metrics], [inputs], [labels])
        paddle.enable_static()
        # step 1: build program
        program_helper.build_program(mode='train')
        program_helper.build_program(mode='eval')
        # support easily to switch mode
        program_helper.to('train')

        forward_ops = program_helper.main_program.block(0).ops
        self.assertEqual(len(forward_ops), 21)

        # step 2: apply optimzer to generate whole program
        optimize_ops, _ = program_helper.apply_optimizer(optimizer)
        all_ops = program_helper.main_program.block(0).ops
        sgd_ops = [
            op for op in program_helper.main_program.block(0).ops
            if op.type == 'sgd'
        ]
        self.assertEqual(len(all_ops), 41)
        self.assertEqual(len(optimize_ops), len(sgd_ops))

        program_helper.reset()


class TestToStatic(unittest.TestCase):

    def test_to_static(self):

        mlp = MLPLayer(hidden_size=hidden_size,
                       intermediate_size=4 * hidden_size,
                       dropout_ratio=0.1,
                       initializer_range=0.02)
        loss = paddle.nn.CrossEntropyLoss()
        optimizer = paddle.optimizer.SGD(learning_rate=0.00001,
                                         parameters=mlp.parameters())

        dataset = MyDataset(batch_num * batch_size)

        inputs = InputSpec([batch_size, hidden_size], 'float32', 'x')
        labels = InputSpec([batch_size], 'int64', 'label')

        engine = Engine(model=mlp,
                        inputs_spec=inputs,
                        labels_spec=labels,
                        strategy=None)
        assert _non_static_mode() == True

        engine.prepare(optimizer=optimizer,
                       loss=loss,
                       metrics=paddle.metric.Accuracy())

        assert _non_static_mode() == False
        engine.fit(dataset, batch_size=batch_size)
        engine.evaluate(dataset, batch_size=batch_size)
        engine.predict(dataset, batch_size=batch_size)


class TestLazyInit(unittest.TestCase):

    def test_lazy_init(self):

        with LazyGuard():
            mlp = MLPLayer(hidden_size=hidden_size,
                           intermediate_size=4 * hidden_size,
                           dropout_ratio=0.1,
                           initializer_range=0.02)
            loss = paddle.nn.CrossEntropyLoss()

        metrics = paddle.metric.Accuracy()
        loss = paddle.nn.CrossEntropyLoss()
        inputs = InputSpec([batch_size, hidden_size], 'float32', 'x')
        labels = InputSpec([batch_size], 'int64', 'label')

        program_helper = ProgramHelper(mlp, loss, [metrics], [inputs], [labels])
        program_helper.build_program(mode='train')
        ops = program_helper.startup_program.block(0).ops
        vars = program_helper.startup_program.block(0).vars
        assert len(vars.keys()) == len(ops)
        program_helper.reset()


if __name__ == "__main__":
    unittest.main()
