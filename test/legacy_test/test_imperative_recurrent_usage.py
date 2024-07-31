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
from paddle import base
from paddle.base import core


class RecurrentTest(paddle.nn.Layer):
    def __init__(self, name_scope):
        super().__init__(name_scope)

    def forward(self, in1, in2):
        out = paddle.matmul(in1, in2)
        sum_out = paddle.sum(out)
        return sum_out, out


class TestRecurrentFeed(unittest.TestCase):
    def test_recurrent_feed(self):
        seed = 90
        original_np1 = np.arange(1, 5).reshape(2, 2).astype("float32")
        original_np2 = np.arange(5, 9).reshape(2, 2).astype("float32")
        with base.dygraph.guard():
            paddle.seed(seed)
            original_in1 = paddle.to_tensor(original_np1)
            original_in2 = paddle.to_tensor(original_np2)
            original_in1.stop_gradient = False
            original_in2.stop_gradient = False
            rt = RecurrentTest("RecurrentTest")

            for i in range(3):
                sum_out, out = rt(original_in1, original_in2)
                out.retain_grads()
                original_in1 = out
                sum_out_value = sum_out.numpy()
                sum_out.backward()
                dyout = out.gradient()
                original_in1.stop_gradient = True
                rt.clear_gradients()

        with base.dygraph.guard():
            paddle.seed(seed)
            original_in1 = paddle.to_tensor(original_np1)
            original_in2 = paddle.to_tensor(original_np2)
            original_in1.stop_gradient = False
            original_in2.stop_gradient = False
            rt = RecurrentTest("RecurrentTest")

            for i in range(3):
                sum_out, out = rt(original_in1, original_in2)
                out.retain_grads()
                original_in1 = out
                eager_sum_out_value = sum_out.numpy()
                sum_out.backward()
                eager_dyout = out.gradient()
                original_in1.stop_gradient = True
                rt.clear_gradients()

        with new_program_scope():
            paddle.seed(seed)
            in1 = paddle.static.data(name="inp1", shape=[2, 2])
            in1.stop_gradient = False
            in2 = paddle.static.data(name="inp2", shape=[2, 2])
            in2.stop_gradient = False
            rt1 = RecurrentTest("RecurrentTest")
            static_sum_out, static_out = rt1(in1, in2)
            static_out.persistable = True
            exe = base.Executor(
                base.CPUPlace()
                if not core.is_compiled_with_cuda()
                else base.CUDAPlace(0)
            )

            if paddle.framework.use_pir_api():
                grad_list = paddle.static.append_backward(static_sum_out)
                _, static_dout = grad_list[-1]
            else:
                base.backward.append_backward(static_sum_out)
                static_dout = (
                    base.default_main_program()
                    .block(0)
                    ._find_var_recursive(static_out.name + "@GRAD")
                )
            fetch_list = [static_sum_out, static_out, static_dout]
            for i in range(3):
                out = exe.run(
                    base.default_main_program(),
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
