#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import os
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core
from paddle.base.param_attr import WeightNormParamAttr

paddle.enable_static()


class TestWeightNormalization(unittest.TestCase):
    batch_size = 3
    hidden_size = 5
    data_desc = (['x', [10], 0],)

    @classmethod
    def setUpClass(cls):
        cls.set_program()

    @classmethod
    def set_program(cls):
        data = paddle.static.data(
            name=cls.data_desc[0][0], shape=[-1] + cls.data_desc[0][1]
        )
        out = paddle.static.nn.fc(
            x=data,
            size=cls.hidden_size,
            weight_attr=WeightNormParamAttr(
                dim=None,
                name='weight_norm_param',
                initializer=paddle.nn.initializer.Constant(1.0),
            ),
            bias_attr=False,
            activation=None,
        )
        loss = paddle.sum(out)
        base.backward.append_backward(loss=loss)
        cls.fetch_list = [
            'weight_norm_param_g',
            'weight_norm_param_v',
            'weight_norm_param_g@GRAD',
        ]

    def run_program(self):
        outputs = []
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(core.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.set_inputs(place)
            exe = base.Executor(place)
            exe.run(base.default_startup_program())
            output = exe.run(
                base.default_main_program(),
                feed=self.inputs,
                fetch_list=self.fetch_list,
                return_numpy=False,
            )
            outputs.append(output)
        self.actual_outputs = outputs

    def set_data(self):
        self.data = collections.OrderedDict()
        for desc in self.data_desc:
            data_name = desc[0]
            data_shape = desc[1]
            data_lod_level = desc[2]
            data_lod = []
            for i in range(data_lod_level):
                lod_level_i = np.random.randint(
                    low=1,
                    high=5,
                    size=(
                        self.batch_size
                        if i == 0
                        else sum(lod_level_i)  # noqa: F821
                    ),
                ).tolist()
                data_lod.append(lod_level_i)
            data_value = np.random.random(
                size=[
                    sum(data_lod[-1]) if data_lod else self.batch_size,
                    *data_shape,
                ]
            ).astype('float32')
            self.data[data_name] = (data_value, data_lod)

    def set_inputs(self, place):
        self.inputs = {}
        for desc in self.data_desc:
            tensor = base.Tensor()
            tensor.set(self.data[desc[0]][0], place)
            self.inputs[desc[0]] = tensor

    def weight_normalize(self):
        v = np.ones(
            (self.data[self.data_desc[0][0]][0].shape[-1], self.hidden_size)
        )
        g = np.linalg.norm(v, axis=None, keepdims=True)
        w = g * v / np.linalg.norm(v, axis=None, keepdims=True)
        x = self.data[self.data_desc[0][0]][0]
        out = np.dot(x, w)
        g_grad = (
            np.dot(x.T, np.ones_like(out))
            * (v / np.linalg.norm(v, axis=None, keepdims=True))
        ).sum(axis=None, keepdims=True)
        return g, v, g_grad

    def test_weight_normalization(self):
        self.set_data()
        self.run_program()
        expect_output = self.weight_normalize()
        for actual_output in self.actual_outputs:
            [
                np.testing.assert_allclose(
                    np.array(actual), expect, rtol=1e-05, atol=0.001
                )
                for expect, actual in zip(expect_output, actual_output)
            ]


if __name__ == '__main__':
    unittest.main()
