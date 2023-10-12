# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import pir
from paddle.base import core
from paddle.framework import LayerHelper

paddle.enable_static()


class TestCastOpTranscriber(unittest.TestCase):
    def test_op(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.to_tensor([2, 3, 4], 'float64')
                y = paddle.cast(x, 'uint8')

        _, mappings = pir.translate_to_new_ir_with_param_map(main_program.desc)
        assert len(str(mappings)) > 0, "no mapping found"


class TestSetValueOp(unittest.TestCase):
    def test_no_mutable_attribute(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        exe = paddle.static.Executor(place)

        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.ones(shape=[2, 3, 4], dtype="float32")
                x = paddle.static.setitem(x, (0, 0), 6)
        ret = exe.run(main_program, fetch_list=x.name)

        x_data = np.ones([2, 3, 4]).astype("float32")
        x_data[0, 0] = 6
        np.testing.assert_array_equal(ret[0], x_data)

    def test_with_mutable_attribute(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        exe = paddle.static.Executor(place)

        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.ones(shape=[2, 3, 4], dtype="float32")
                zero = paddle.full([], 0, dtype="int32")
                x = paddle.static.setitem(x, zero, 6)
        ret = exe.run(main_program, fetch_list=x.name)

        x_data = np.ones([2, 3, 4]).astype("float32")
        x_data[0] = 6
        np.testing.assert_array_equal(ret[0], x_data)

    def test_grad(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        exe = paddle.static.Executor(place)
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        input_shape = [7, 6, 5, 4, 3, 2]
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.ones(shape=input_shape, dtype="float32")
                value = paddle.tensor.fill_constant([1, 3, 2], "float32", 1)
                # test stop_gradient
                value.stop_gradient = False
                x.stop_gradient = False
                attrs = {
                    'axes': [0],
                    'starts': [6],
                    'ends': [0],
                    'steps': [-4],
                    'decrease_axes': [],
                    'none_axes': [],
                    'dtype': paddle.float32,
                }
                inputs = {'Input': x, 'ValueTensor': value}

                helper = LayerHelper("set_value")
                y = helper.create_variable_for_type_inference(dtype=x.dtype)

                helper.append_op(
                    type="set_value",
                    inputs=inputs,
                    outputs={'Out': y},
                    attrs=attrs,
                )
                y2 = y + 1
                loss = paddle.sum(y2)
                opt = paddle.optimizer.Adam()
                opt.minimize(loss)

                x_data = np.arange(
                    0, np.prod(input_shape), dtype="float32"
                ).reshape(input_shape)
                fetch_list = [x.grad_name, value.grad_name]
                ret = exe.run(main_program, fetch_list=fetch_list)
                self.assertTrue((ret[0][6:0:-4] == 0).all())


class TestCheckUnregisteredOp(unittest.TestCase):
    def test_program(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.randn((4, 16))
            prev_h = paddle.randn((4, 32))

            cell = paddle.nn.SimpleRNNCell(16, 32)
            y, h = cell(x, prev_h)

        ops = pir.check_unregistered_ops(main_program.desc)
        assert len(ops) == 0


if __name__ == "__main__":
    unittest.main()
