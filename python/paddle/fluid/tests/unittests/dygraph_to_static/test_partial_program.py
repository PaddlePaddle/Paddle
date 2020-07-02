# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.layers.utils import flatten
from paddle.fluid.dygraph import declarative

import unittest

SEED = 2020


def nested_input(x, y):
    sum_res = x + y[0]

    z_elem = y[3]['z']
    sub_res = z_elem[0] - z_elem[1]

    mul_res = y[-1]['d']['da'] * y[-1]['d']['dc']
    mean_func = fluid.layers.mean
    out = mean_func(sub_res) + mean_func(sum_res) + mean_func(mul_res)
    return out


def nested_output(x, y):
    sum_res = x + y
    sub_res = x - y
    mul_res = x * y

    out = {}
    out['z'] = sum_res
    out['a'] = [sub_res, 64, [mul_res, "cmd"]]
    return out


def fake_data(shape):
    x_data = np.random.random(shape).astype('float32')
    return fluid.dygraph.to_variable(x_data)


class TestWithNestedInput(unittest.TestCase):
    def setUp(self):
        self.x = None
        self.y = None

    def fake_input(self):
        self.x = fake_data([10, 16])
        self.y = [
            fake_data([10, 16]), "preprocess_cmd", 64, {
                'z': [fake_data([10, 12]), fake_data([10, 12])],
                'c': fake_data([10, 10]),
                'd': {
                    'da': 12,
                    'dc': fake_data([10, 10])
                }
            }
        ]

    def _run(self, to_static):
        with fluid.dygraph.guard():
            if self.x is None or self.y is None:
                self.fake_input()

            if to_static:
                out = declarative(nested_input)(self.x, self.y)
            else:
                out = nested_input(self.x, self.y)

        return out.numpy()

    def test_nest(self):
        dygraph_res = self._run(to_static=False)
        static_res = self._run(to_static=True)
        self.assertTrue(np.allclose(dygraph_res, static_res))


class TestWithNestedOutput(unittest.TestCase):
    def setUp(self):
        self.x = None
        self.y = None

    def _run(self, to_static):
        with fluid.dygraph.guard():
            if self.x is None or self.y is None:
                self.x = fake_data([10, 16])
                self.y = fake_data([10, 16])

            if to_static:
                out = declarative(nested_output)(self.x, self.y)
            else:
                out = nested_output(self.x, self.y)

        return out

    def test_nest(self):
        dygraph_res = self._run(to_static=False)
        dygraph_res = flatten(dygraph_res)

        static_res = self._run(to_static=True)
        static_res = flatten(static_res)

        self.assertTrue(len(dygraph_res) == len(static_res))

        for dy_var, st_var in zip(dygraph_res, static_res):
            if isinstance(dy_var, fluid.core.VarBase):
                self.assertTrue(np.allclose(dy_var.numpy(), st_var.numpy()))
            else:
                self.assertTrue(dy_var, st_var)


if __name__ == '__main__':
    unittest.main()
