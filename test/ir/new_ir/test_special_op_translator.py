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
from paddle import ir
from paddle.fluid import core
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

        _ = ir.translate_to_new_ir(main_program.desc)


class TestElementwiseOpTranscriber(unittest.TestCase):
    def test_elementwise_without_y_grad(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        exe = paddle.static.Executor(place)

        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x_data = np.random.rand(100, 2, 3)
                y_data = np.random.rand(100)
                x = paddle.to_tensor(x_data, dtype='float32')
                x.stop_gradient = False
                y = paddle.to_tensor(y_data, dtype='float32')

                out1 = paddle.tensor.math._elementwise_op(
                    LayerHelper('elementwise_add', x=x, y=y, axis=0)
                )
                out1.stop_gradient = False
                mean = paddle.mean(out1)
                paddle.static.append_backward(mean)

                out = exe.run(main_program, {}, fetch_list=[out1.name])
                np.testing.assert_allclose(
                    out[0],
                    x_data + y_data.reshape(100, 1, 1),
                    rtol=1e-6,
                    atol=1e-6,
                )

    def test_elementwise_with_y_grad(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        exe = paddle.static.Executor(place)

        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x_data = np.random.rand(100, 2, 3)
                y_data = np.random.rand(100)
                x = paddle.to_tensor(x_data, dtype='float32')
                x.stop_gradient = False
                y = paddle.to_tensor(y_data, dtype='float32')
                y.stop_gradient = False

                out1 = paddle.tensor.math._elementwise_op(
                    LayerHelper('elementwise_add', x=x, y=y, axis=0)
                )
                out1.stop_gradient = False
                mean = paddle.mean(out1)
                paddle.static.append_backward(mean)

                out = exe.run(main_program, {}, fetch_list=[out1.name])
                np.testing.assert_allclose(
                    out[0],
                    x_data + y_data.reshape(100, 1, 1),
                    rtol=1e-6,
                    atol=1e-6,
                )


class TestEmbeddingOpTranscriber(unittest.TestCase):
    def test_op(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.static.data(name="x", shape=[2, 4], dtype=np.int64)
                embedding = paddle.nn.Embedding(
                    10, 3, weight_attr=paddle.nn.initializer.Constant(value=1.0)
                )
                output = embedding(x)

        _ = ir.translate_to_new_ir(main_program.desc)


class TestIncrementOpTranscriber(unittest.TestCase):
    def test_op(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                data = paddle.zeros(shape=[1], dtype='float32')
                counter = paddle.increment(data)

        _ = ir.translate_to_new_ir(main_program.desc)


class TestAssignValueOpTranscriber(unittest.TestCase):
    def test_op(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.to_tensor(
                    [[0.1, 0.2], [0.3, 0.4]],
                    place=paddle.CPUPlace(),
                    stop_gradient=False,
                )

        _ = ir.translate_to_new_ir(main_program.desc)


class TestRnnOpTranscriber(unittest.TestCase):
    def test_op(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.randn((4, 16))
                prev_h = paddle.randn((4, 32))

                cell = paddle.nn.SimpleRNNCell(16, 32)
                y, h = cell(x, prev_h)

        _ = ir.translate_to_new_ir(main_program.desc)


class TestEmptyVarTranslate(unittest.TestCase):
    def test_op(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x1 = paddle.rand(shape=[3, 3], dtype="float32")
                x1.stop_gradient = False
                weight = paddle.full(
                    shape=[3, 3], fill_value="0.5", dtype="float32"
                )
                y = paddle.nn.functional.linear(x1, weight)
                y.stop_gradient = True
                out1 = paddle.concat(x=[x1, y], axis=1)
                out2 = paddle.mean(out1)
                sgd_optimizer = paddle.optimizer.SGD(learning_rate=0.1)
                sgd_optimizer.minimize(out2)
        _ = ir.translate_to_new_ir(main_program.desc)


class TestOneHotOpTranscriber(unittest.TestCase):
    def test_mutable_attribute(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                depth = paddle.assign(np.array([10], dtype=np.int32))
                label = paddle.static.data(
                    name="label", shape=[-1, 1], dtype="int64"
                )
                one_hot_label = paddle.nn.functional.one_hot(
                    x=label, num_classes=depth
                )

        _ = ir.translate_to_new_ir(main_program.desc)

    def test_normal_attribute(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                depth = 10
                label = paddle.static.data(
                    name="label", shape=[-1, 1], dtype="int64"
                )
                one_hot_label = paddle.nn.functional.one_hot(
                    x=label, num_classes=depth
                )

        _ = ir.translate_to_new_ir(main_program.desc)


class TestReduceOpTranscriber(unittest.TestCase):
    def test_reduce_all(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        exe = paddle.static.Executor(place)

        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                arr = np.ones([2, 2], dtype="float32")
                x = paddle.to_tensor(arr, dtype='int32')
                out1 = paddle.all(x)

                out = exe.run(main_program, {}, fetch_list=[out1.name])
                np.testing.assert_array_equal(out[0], np.all(arr))

    def test_with_axis(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        exe = paddle.static.Executor(place)

        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                arr = np.ones([2, 2], dtype="float32")
                x = paddle.to_tensor(arr, dtype='int32')
                out1 = paddle.all(x, axis=0)

                out = exe.run(main_program, {}, fetch_list=[out1.name])
                np.testing.assert_array_equal(out[0], np.all(arr, axis=0))


class TestIndexPutOpTranscriber(unittest.TestCase):
    def test_op(self):
        place = core.Place()
        place.set_place(paddle.CPUPlace())
        new_scope = paddle.static.Scope()
        main_program = paddle.static.Program()
        with paddle.static.scope_guard(new_scope):
            with paddle.static.program_guard(main_program):
                x = paddle.randn([2, 3])
                indices = [paddle.randint(0, 2, [2]), paddle.randint(0, 1, [2])]
                value = paddle.randn([2])
                y = paddle.index_put(x, indices, value, False)

        _ = ir.translate_to_new_ir(main_program.desc)


if __name__ == "__main__":
    unittest.main()
