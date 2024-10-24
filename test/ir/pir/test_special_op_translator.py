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
        with paddle.pir_utils.OldIrGuard():
            place = core.Place()
            place.set_place(paddle.CPUPlace())
            new_scope = paddle.static.Scope()
            main_program = paddle.static.Program()
            with paddle.static.scope_guard(new_scope):
                with paddle.static.program_guard(main_program):
                    x = paddle.to_tensor([2, 3, 4], 'float64')
                    y = paddle.cast(x, 'uint8')

            _, mappings = pir.translate_to_pir_with_param_map(main_program.desc)
            assert len(str(mappings)) > 0, "no mapping found"


class TestCondWithInplace(unittest.TestCase):
    def test_op(self):
        with paddle.pir_utils.OldIrGuard():

            def cond_with_inplace():
                x = paddle.ones(shape=[2, 1, 2, 3], dtype="float32")
                y = paddle.ones(shape=[2, 1, 2, 3], dtype="float32")
                running_mean = paddle.to_tensor([0], dtype="float32")
                running_variance = paddle.to_tensor([1], dtype="float32")
                weight = paddle.to_tensor([2], dtype="float32")
                bias = paddle.to_tensor([1], dtype="float32")
                y = paddle.nn.functional.batch_norm(
                    x, running_mean, running_variance, weight, bias
                )

            legacy_program = paddle.jit.to_static(
                cond_with_inplace,
                input_spec=[],
                full_graph=True,
            )

            l = pir.translate_to_pir(legacy_program.main_program.desc)
            assert l is not None

    def test_nested_op(self):
        with paddle.pir_utils.OldIrGuard():

            def cond_with_inplace():
                x = paddle.ones(shape=[2, 1, 2, 3], dtype="float32")
                y = paddle.ones(shape=[2, 1, 2, 3], dtype="float32")
                z = paddle.ones(shape=[2, 1, 2, 3], dtype="float32")
                running_mean = paddle.to_tensor([0], dtype="float32")
                running_variance = paddle.to_tensor([1], dtype="float32")
                weight = paddle.to_tensor([2], dtype="float32")
                bias = paddle.to_tensor([1], dtype="float32")
                if y > z:
                    z = paddle.nn.functional.batch_norm(
                        z, running_mean, running_variance, weight, bias
                    )
                else:
                    y = paddle.nn.functional.batch_norm(
                        x, running_mean, running_variance, weight, bias
                    )

            legacy_program = paddle.jit.to_static(
                cond_with_inplace,
                input_spec=[],
                full_graph=True,
            )

            l = pir.translate_to_pir(legacy_program.main_program.desc)
            assert l is not None


class TestElementwiseOpTranscriber(unittest.TestCase):
    def test_elementwise_without_y_grad(self):
        with paddle.pir_utils.OldIrGuard():
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

                    out = exe.run(main_program, {}, fetch_list=[out1])
                    np.testing.assert_allclose(
                        out[0],
                        x_data + y_data.reshape(100, 1, 1),
                        rtol=1e-6,
                        atol=1e-6,
                    )

    def test_elementwise_with_y_grad(self):
        with paddle.pir_utils.OldIrGuard():
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

                    out = exe.run(main_program, {}, fetch_list=[out1])
                    np.testing.assert_allclose(
                        out[0],
                        x_data + y_data.reshape(100, 1, 1),
                        rtol=1e-6,
                        atol=1e-6,
                    )

    def test_add_inplace(self):
        with paddle.pir_utils.OldIrGuard():
            place = core.Place()
            place.set_place(paddle.CPUPlace())
            exe = paddle.static.Executor(place)

            new_scope = paddle.static.Scope()
            main_program = paddle.static.Program()
            with paddle.static.scope_guard(new_scope):
                with paddle.static.program_guard(main_program):
                    x = paddle.ones(shape=(100, 2, 3), dtype='float32')
                    y = paddle.ones(shape=(100, 2, 3), dtype='float32')

                    helper = LayerHelper('elementwise_add')
                    helper.append_op(
                        type="elementwise_add",
                        inputs={"X": x, "Y": y},
                        outputs={"Out": y},
                        attrs={"axis": -1},
                    )
            _ = pir.translate_to_pir(main_program.desc)


class TestEmbeddingOpTranscriber(unittest.TestCase):
    def test_op(self):
        with paddle.pir_utils.OldIrGuard():
            place = core.Place()
            place.set_place(paddle.CPUPlace())
            new_scope = paddle.static.Scope()
            main_program = paddle.static.Program()
            with paddle.static.scope_guard(new_scope):
                with paddle.static.program_guard(main_program):
                    x = paddle.static.data(
                        name="x", shape=[2, 4], dtype=np.int64
                    )
                    embedding = paddle.nn.Embedding(
                        10,
                        3,
                        weight_attr=paddle.nn.initializer.Constant(value=1.0),
                    )
                    output = embedding(x)

            _ = pir.translate_to_pir(main_program.desc)


class TestIncrementOpTranscriber(unittest.TestCase):
    def test_op(self):
        with paddle.pir_utils.OldIrGuard():
            place = core.Place()
            place.set_place(paddle.CPUPlace())
            new_scope = paddle.static.Scope()
            main_program = paddle.static.Program()
            with paddle.static.scope_guard(new_scope):
                with paddle.static.program_guard(main_program):
                    data = paddle.zeros(shape=[1], dtype='float32')
                    counter = paddle.increment(data)

            _ = pir.translate_to_pir(main_program.desc)


class TestAssignValueOpTranscriber(unittest.TestCase):
    def test_op(self):
        with paddle.pir_utils.OldIrGuard():
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

            _ = pir.translate_to_pir(main_program.desc)


class TestRnnOpTranscriber(unittest.TestCase):
    def test_op(self):
        with paddle.pir_utils.OldIrGuard():
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

            _ = pir.translate_to_pir(main_program.desc)


class TestEmptyVarTranslate(unittest.TestCase):
    def test_op(self):
        with paddle.pir_utils.OldIrGuard():
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
            _ = pir.translate_to_pir(main_program.desc)


class TestOneHotOpTranscriber(unittest.TestCase):
    def test_mutable_attribute(self):
        with paddle.pir_utils.OldIrGuard():
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

            _ = pir.translate_to_pir(main_program.desc)

    def test_normal_attribute(self):
        with paddle.pir_utils.OldIrGuard():
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

            _ = pir.translate_to_pir(main_program.desc)


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

                out = exe.run(main_program, {}, fetch_list=[out1])
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

                out = exe.run(main_program, {}, fetch_list=[out1])
                np.testing.assert_array_equal(out[0], np.all(arr, axis=0))


class TestIndexPutOpTranscriber(unittest.TestCase):
    def test_op(self):
        with paddle.pir_utils.OldIrGuard():
            place = core.Place()
            place.set_place(paddle.CPUPlace())
            new_scope = paddle.static.Scope()
            main_program = paddle.static.Program()
            with paddle.static.scope_guard(new_scope):
                with paddle.static.program_guard(main_program):
                    x = paddle.randn([2, 3])
                    indices = [
                        paddle.randint(0, 2, [2]),
                        paddle.randint(0, 1, [2]),
                    ]
                    value = paddle.randn([2])
                    y = paddle.index_put(x, indices, value, False)

            _ = pir.translate_to_pir(main_program.desc)


class TestGradAddOpTranscriber(unittest.TestCase):
    def test_op(self):
        with paddle.pir_utils.OldIrGuard():
            place = core.Place()
            place.set_place(paddle.CPUPlace())
            new_scope = paddle.static.Scope()
            main_program = paddle.static.Program()
            with paddle.static.scope_guard(new_scope):
                with paddle.static.program_guard(main_program):
                    x_data = np.random.rand(100, 2, 3)
                    y_data = np.random.rand(100, 1, 1)
                    x = paddle.to_tensor(x_data, dtype='float32')
                    x.stop_gradient = False
                    y = paddle.to_tensor(y_data, dtype='float32')

                    helper = LayerHelper('grad_add')
                    out = helper.create_variable_for_type_inference("float")
                    helper.append_op(
                        type="grad_add",
                        inputs={"X": x, "Y": y},
                        outputs={"Out": out},
                        attrs={"axis": -1},
                    )

            _ = pir.translate_to_pir(main_program.desc)


class TestShadowOutputSlice(unittest.TestCase):
    def test_op(self):
        with paddle.pir_utils.OldIrGuard():
            place = core.Place()
            place.set_place(paddle.CPUPlace())
            new_scope = paddle.static.Scope()
            main_program = paddle.static.Program()
            with paddle.static.scope_guard(new_scope):
                with paddle.static.program_guard(main_program):
                    x = paddle.rand([3, 9, 5])
                    y = paddle.static.data(
                        name="y", shape=[3, 9, 5], dtype="float32"
                    )

                    _, out, _ = paddle.split(x, num_or_sections=3, axis=1)
                    helper = LayerHelper('shadow_output')
                    helper.append_op(
                        type="shadow_output",
                        inputs={"x": [out.name]},
                        outputs={"out": [y.name]},
                        attrs={"name": out.name},
                    )

            l = pir.translate_to_pir(main_program.desc)


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
        ret = exe.run(main_program, fetch_list=[x])

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
        ret = exe.run(main_program, fetch_list=[x])

        x_data = np.ones([2, 3, 4]).astype("float32")
        x_data[0] = 6
        np.testing.assert_array_equal(ret[0], x_data)

    def test_grad(self):
        with paddle.pir_utils.OldIrGuard():
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


class TestShareBufferOpTranscriber(unittest.TestCase):
    def test_program(self):
        with paddle.pir_utils.OldIrGuard():
            place = core.Place()
            place.set_place(paddle.CPUPlace())

            new_scope = paddle.static.Scope()
            main_program = paddle.static.Program()
            with paddle.static.scope_guard(new_scope):
                with paddle.static.program_guard(main_program):
                    x = paddle.ones(shape=(100, 2, 3), dtype='float32')
                    y = paddle.ones(shape=(100, 2, 3), dtype='float32')

                    helper = LayerHelper('share_buffer')
                    helper.append_op(
                        type="share_buffer",
                        inputs={"X": x},
                        outputs={"Out": y, "XOut": x},
                    )
            l = pir.translate_to_pir(main_program.desc)
            assert (
                l.global_block().ops[2].name() == "pd_op.share_data_"
            ), "share_buffer should be translated to share_data_"


class TestDataOp(unittest.TestCase):
    def test_data_op(self):
        with paddle.pir_utils.OldIrGuard():
            place = core.Place()
            place.set_place(paddle.CPUPlace())

            new_scope = paddle.static.Scope()
            main_program = paddle.static.Program()
            with paddle.static.scope_guard(new_scope):
                with paddle.static.program_guard(main_program):
                    _ = paddle.static.data(
                        name="y", shape=[3, 9, 5], dtype="int64"
                    )
            l = pir.translate_to_pir(main_program.desc)
            self.assertTrue(len(l.global_block().ops) > 0)
            self.assertTrue(l.global_block().ops[0].name() == "pd_op.data")
            data_op = l.global_block().ops[0]
            self.assertIn("dtype", data_op.attrs())
            self.assertEqual(str(data_op.attrs()["dtype"]), "paddle.int64")


class TestCheckUnregisteredOp(unittest.TestCase):
    def test_program(self):
        with paddle.pir_utils.OldIrGuard():
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
