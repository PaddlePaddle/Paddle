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

import unittest

import numpy as np
from test_imperative_base import new_program_scope

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.dygraph.base import to_variable


class RecurrentTest(fluid.Layer):
    def __init__(self, name_scope):
        super().__init__(name_scope)

    def forward(self, in1, in2):
        out = fluid.layers.mul(in1, in2)
        sum_out = paddle.sum(out)
        return sum_out, out


class TestRecurrentFeed(unittest.TestCase):
    def test_recurrent_feed(self):

        seed = 90
        original_np1 = np.arange(1, 5).reshape(2, 2).astype("float32")
        original_np2 = np.arange(5, 9).reshape(2, 2).astype("float32")
        with fluid.dygraph.guard():
            fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            original_in1 = to_variable(original_np1)
            original_in2 = to_variable(original_np2)
            original_in1.stop_gradient = False
            original_in2.stop_gradient = False
            rt = RecurrentTest("RecurrentTest")

            for i in range(3):
                sum_out, out = rt(original_in1, original_in2)
                original_in1 = out
                sum_out_value = sum_out.numpy()
                sum_out.backward()
                dyout = out.gradient()
                original_in1.stop_gradient = True
                rt.clear_gradients()
            fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})

        with fluid.dygraph.guard():
            fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            original_in1 = to_variable(original_np1)
            original_in2 = to_variable(original_np2)
            original_in1.stop_gradient = False
            original_in2.stop_gradient = False
            rt = RecurrentTest("RecurrentTest")

            for i in range(3):
                sum_out, out = rt(original_in1, original_in2)
                original_in1 = out
                eager_sum_out_value = sum_out.numpy()
                sum_out.backward()
                eager_dyout = out.gradient()
                original_in1.stop_gradient = True
                rt.clear_gradients()
            fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": False})

        with new_program_scope():
            fluid.default_startup_program().random_seed = seed
            fluid.default_main_program().random_seed = seed
            in1 = fluid.layers.data(
                name="inp1", shape=[2, 2], append_batch_size=False
            )
            in2 = fluid.layers.data(
                name="inp2", shape=[2, 2], append_batch_size=False
            )
            rt1 = RecurrentTest("RecurrentTest")
            static_sum_out, static_out = rt1(in1, in2)
            fluid.backward.append_backward(static_sum_out)
            exe = fluid.Executor(
                fluid.CPUPlace()
                if not core.is_compiled_with_cuda()
                else fluid.CUDAPlace(0)
            )

            static_dout = (
                fluid.default_main_program()
                .block(0)
                ._find_var_recursive(static_out.name + "@GRAD")
            )
            fetch_list = [static_sum_out, static_out, static_dout]
            for i in range(3):
                out = exe.run(
                    fluid.default_main_program(),
                    feed={"inp1": original_np1, "inp2": original_np2},
                    fetch_list=fetch_list,
                )
                static_out_value = out[1]
                static_sum_out = out[0]
                static_dout = out[2]
                original_np1 = static_out_value

        np.testing.assert_array_equal(static_sum_out, sum_out_value)
        np.testing.assert_array_equal(static_sum_out, eager_sum_out_value)
        np.testing.assert_array_equal(static_dout, dyout)
        np.testing.assert_array_equal(static_dout, eager_dyout)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
