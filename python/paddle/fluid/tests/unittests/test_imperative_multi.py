# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from paddle.fluid.imperative import Embedding, LayerNorm, FC, to_variable, Layer, guard
from test_imperative_base import new_program_scope
from paddle.fluid import core
import numpy as np
import six
import pdb
np.set_printoptions(suppress=True)


class Multi_Input(Layer):
    def __init__(self, name_scope, d_key, n_head):
        super(Multi_Input, self).__init__(name_scope)
        self._q_fc = FC(name_scope=self.full_name(),
                        size=d_key * n_head,
                        bias_attr=False)
        self._k_fc = FC(name_scope=self.full_name(),
                        size=d_key * n_head,
                        bias_attr=False)
        self._v_fc = FC(name_scope=self.full_name(),
                        size=d_key * n_head,
                        bias_attr=False)

    def forward(self, input):
        self.i = fluid.layers.reshape(input, shape=[4, 3])
        k = self._k_fc(self.i)
        # q = self._q_fc(self.i)
        # v = self._v_fc(self.i)

        # tmp = fluid.layers.concat([k, q, v], axis=1)
        tmp = fluid.layers.concat([k], axis=1)
        return tmp


class MLP(fluid.imperative.Layer):
    def __init__(self, name_scope):
        super(MLP, self).__init__(name_scope)
        self._fc1 = FC(self.full_name(), 3, bias_attr=False)
        self._fc2 = FC(self.full_name(), 4, bias_attr=False)
        self._fc3 = FC(self.full_name(), 4, bias_attr=False)
        self._fc4 = FC(self.full_name(), 4, bias_attr=False)

        # self._fc2 = FC(self.full_name(),
        #                4,
        #                param_attr=fluid.ParamAttr(
        #                    initializer=fluid.initializer.Constant(value=0.1)),
        #                bias_attr=fluid.ParamAttr(
        #                    initializer=fluid.initializer.Constant(value=0.1)))
        # self._fc3 = FC(self.full_name(),
        #                4,                        param_attr=fluid.ParamAttr(
        #         initializer=fluid.initializer.Constant(value=0.1)),
        #                bias_attr=fluid.ParamAttr(
        #                    initializer=fluid.initializer.Constant(value=0.1)))

    def forward(self, inputs):
        x = self._fc1(inputs)
        m = x
        # if fluid.framework._in_imperative_mode():
        #     x1 = to_variable(x)
        # else:
        #     x1 = fluid.layers.assign(x)
        y1 = self._fc2(x)
        y2 = self._fc3(x)
        y3 = self._fc4(x)
        cc = fluid.layers.concat([y1, y2, y3], axis=0)
        rlt = fluid.layers.reduce_sum(cc)
        return rlt, m


class TestMulti_Input(unittest.TestCase):
    def test(self):
        seed = 90
        np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        batch_num = 1
        with fluid.imperative.guard():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            var_inp = fluid.imperative.base.to_variable(np_inp)
            mlp = MLP("mlp2")
            optimizer = fluid.optimizer.SGD(learning_rate=0.003)
            for i in range(batch_num):
                out, m = mlp(var_inp)
                out_np = out._numpy()
                out._backward()
                # optimizer.minimize(out)
                # mlp.clear_gradients()
                dm_grad = m._gradient()
                print(m.name)

        with new_program_scope():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            inp = fluid.layers.data(
                name="inp", shape=[2, 2], append_batch_size=False)
            mlp = MLP("mlp2")
            static_out, m = mlp(inp)

            optimizer = fluid.optimizer.SGD(learning_rate=0.003)
            # optimizer.minimize(static_out)
            fluid.backward.append_backward(static_out)
            exe = fluid.Executor(fluid.CPUPlace(
            ) if not core.is_compiled_with_cuda() else fluid.CUDAPlace(0))
            exe.run(fluid.default_startup_program())
            # print(fluid.framework.default_main_program().block(0).ops)
            for i in range(batch_num):
                st_out, st_grad = exe.run(
                    feed={inp.name: np_inp},
                    fetch_list=[static_out.name, m.name + '@GRAD'])

        print(out_np)
        print("==============")
        print(st_out)
        self.assertTrue(np.array_equal(dm_grad, st_grad))


if __name__ == '__main__':
    unittest.main()
