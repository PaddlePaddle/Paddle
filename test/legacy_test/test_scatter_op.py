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

import os
import unittest

import numpy as np
from eager_op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import base
from paddle.base import core
from paddle.base.dygraph.base import switch_to_static_graph


class TestScatterOp(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        self.public_python_api = paddle.scatter
        self.prim_op_type = "prim"
        self._set_dtype()
        self.if_enable_cinn()
        target_dtype = "float16" if self.dtype == np.float16 else "float32"
        ref_np = np.ones((3, 50)).astype(target_dtype)
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 50)).astype(target_dtype)
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        if self.dtype == np.uint16:
            ref_np = convert_float_to_uint16(ref_np)
            updates_np = convert_float_to_uint16(updates_np)
            output_np = convert_float_to_uint16(output_np)
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def if_enable_cinn(self):
        pass

    def _set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X", "Updates"], "Out", check_prim=True)


class TestScatterFP16Op(TestScatterOp):
    def _set_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestScatterBF16Op(TestScatterOp):
    def _set_dtype(self):
        self.dtype = np.uint16

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['X', 'Updates'],
                'Out',
                check_prim=True,
            )


class TestScatterOp0(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        self.public_python_api = paddle.scatter
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        self._set_dtype()
        target_dtype = "float16" if self.dtype == np.float16 else "float32"
        ref_np = np.ones((3, 3)).astype(target_dtype)
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 3)).astype(target_dtype)
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        if self.dtype == np.uint16:
            ref_np = convert_float_to_uint16(ref_np)
            updates_np = convert_float_to_uint16(updates_np)
            output_np = convert_float_to_uint16(output_np)
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.attrs = {'overwrite': True}
        self.outputs = {'Out': output_np}

    def if_enable_cinn(self):
        pass

    def _set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X", "Updates"], "Out", check_prim=True)


class TestScatterFP16Op0(TestScatterOp0):
    def _set_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestScatterBF16Op0(TestScatterOp0):
    def _set_dtype(self):
        self.dtype = np.uint16

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['X', 'Updates'],
                'Out',
                check_prim=True,
            )


class TestScatterOp1(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        self.public_python_api = paddle.scatter
        self.prim_op_type = "prim"
        self._set_dtype()
        self.if_enable_cinn()
        target_dtype = "float16" if self.dtype == np.float16 else "float32"
        ref_np = np.ones((3, 3)).astype(target_dtype)
        zeros_np = np.zeros([2, 3]).astype(target_dtype)
        index_np = np.array([1, 1]).astype("int32")
        updates_np = np.random.random((2, 3)).astype(target_dtype)
        output_np = np.copy(ref_np)
        output_np[index_np] = zeros_np
        for i in range(0, len(index_np)):
            output_np[index_np[i]] += updates_np[i]
        if self.dtype == np.uint16:
            ref_np = convert_float_to_uint16(ref_np)
            updates_np = convert_float_to_uint16(updates_np)
            output_np = convert_float_to_uint16(output_np)
        self.attrs = {'overwrite': False}
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def if_enable_cinn(self):
        pass

    def _set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X", "Updates"], "Out", check_prim=True)


class TestScatterFP16Op1(TestScatterOp1):
    def _set_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestScatterBF16Op1(TestScatterOp1):
    def _set_dtype(self):
        self.dtype = np.uint16

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['X', 'Updates'],
                'Out',
                check_prim=True,
            )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestScatterOp2(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        self.public_python_api = paddle.scatter
        self.prim_op_type = "prim"
        self._set_dtype()
        self.if_enable_cinn()
        target_dtype = "float16" if self.dtype == np.float16 else "float32"
        ref_np = np.ones((3, 3)).astype(target_dtype)
        index_np = np.array([1, 2]).astype("int32")
        updates_np = np.random.random((2, 3)).astype(target_dtype)
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        if self.dtype == np.uint16:
            ref_np = convert_float_to_uint16(ref_np)
            updates_np = convert_float_to_uint16(updates_np)
            output_np = convert_float_to_uint16(output_np)
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def if_enable_cinn(self):
        pass

    def _set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-3)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['X', 'Updates'],
                'Out',
                check_prim=True,
            )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestScatterFP16Op2(TestScatterOp2):
    def _set_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestScatterBF16Op2(TestScatterOp2):
    def _set_dtype(self):
        self.dtype = np.uint16

    def if_enable_cinn(self):
        self.enable_cinn = False


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestScatterOp3(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        self.public_python_api = paddle.scatter
        self.prim_op_type = "prim"
        self._set_dtype()
        self.if_enable_cinn()
        target_dtype = "float16" if self.dtype == np.float16 else "float32"
        ref_np = np.ones((3, 3)).astype(target_dtype)
        zeros_np = np.zeros([2, 3]).astype(target_dtype)
        index_np = np.array([1, 1]).astype("int32")
        updates_np = np.random.random((2, 3)).astype(target_dtype)
        output_np = np.copy(ref_np)
        output_np[index_np] = zeros_np
        for i in range(0, len(index_np)):
            output_np[index_np[i]] += updates_np[i]
        if self.dtype == np.uint16:
            ref_np = convert_float_to_uint16(ref_np)
            updates_np = convert_float_to_uint16(updates_np)
            output_np = convert_float_to_uint16(output_np)
        self.attrs = {'overwrite': False}
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def if_enable_cinn(self):
        pass

    def _set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-3)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['X', 'Updates'],
                'Out',
                check_prim=True,
            )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestScatterFP16Op3(TestScatterOp3):
    def _set_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestScatterBF16Op3(TestScatterOp3):
    def _set_dtype(self):
        self.dtype = np.uint16

    def if_enable_cinn(self):
        self.enable_cinn = False


class TestScatterOp4(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        self.public_python_api = paddle.scatter
        self.prim_op_type = "prim"
        self._set_dtype()
        self.if_enable_cinn()
        target_dtype = "float16" if self.dtype == np.float16 else "float32"
        ref_np = np.ones((3, 3)).astype(target_dtype)
        index_np = np.array([1, 2]).astype("int64")
        updates_np = np.random.random((2, 3)).astype(target_dtype)
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        if self.dtype == np.uint16:
            ref_np = convert_float_to_uint16(ref_np)
            updates_np = convert_float_to_uint16(updates_np)
            output_np = convert_float_to_uint16(output_np)
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def if_enable_cinn(self):
        pass

    def _set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Updates'], 'Out', check_prim=True)


class TestScatterFP16Op4(TestScatterOp4):
    def _set_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestScatterBF16Op4(TestScatterOp4):
    def _set_dtype(self):
        self.dtype = np.uint16

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['X', 'Updates'],
                'Out',
                check_prim=True,
            )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestScatterOp5(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        self.public_python_api = paddle.scatter
        self.prim_op_type = "prim"
        self._set_dtype()
        self.if_enable_cinn()
        target_dtype = "float16" if self.dtype == np.float16 else "float32"
        ref_np = np.ones((3, 3)).astype(target_dtype)
        index_np = np.array([1, 2]).astype("int64")
        updates_np = np.random.random((2, 3)).astype(target_dtype)
        output_np = np.copy(ref_np)
        output_np[index_np] = updates_np
        if self.dtype == np.uint16:
            ref_np = convert_float_to_uint16(ref_np)
            updates_np = convert_float_to_uint16(updates_np)
            output_np = convert_float_to_uint16(output_np)
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def if_enable_cinn(self):
        pass

    def _set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-3)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['X', 'Updates'],
                'Out',
                check_prim=True,
            )


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestScatterFP16Op5(TestScatterOp5):
    def _set_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestScatterBF16Op5(TestScatterOp5):
    def _set_dtype(self):
        self.dtype = np.uint16

    def if_enable_cinn(self):
        self.enable_cinn = False


class TestScatterOp6(OpTest):
    def setUp(self):
        self.op_type = "scatter"
        self.python_api = paddle.scatter
        self.public_python_api = paddle.scatter
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        self._set_dtype()
        target_dtype = "float16" if self.dtype == np.float16 else "float32"
        ref_np = np.ones((3, 50)).astype(target_dtype)
        index_np = np.array([[1], [2]]).astype("int32")
        updates_np = np.random.random((2, 50)).astype(target_dtype)
        output_np = np.copy(ref_np)
        output_np[np.array([1, 2]).astype("int32")] = updates_np
        if self.dtype == np.uint16:
            ref_np = convert_float_to_uint16(ref_np)
            updates_np = convert_float_to_uint16(updates_np)
            output_np = convert_float_to_uint16(output_np)
        self.inputs = {'X': ref_np, 'Ids': index_np, 'Updates': updates_np}
        self.outputs = {'Out': output_np}

    def if_enable_cinn(self):
        pass

    def _set_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X", "Updates"], "Out", check_prim=True)


class TestScatterFP16Op6(TestScatterOp6):
    def _set_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestScatterBF16Op6(TestScatterOp6):
    def if_enable_cinn(self):
        self.enable_cinn = False

    def _set_dtype(self):
        self.dtype = np.uint16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place)

    def test_check_grad(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['X', 'Updates'],
                'Out',
                check_prim=True,
            )


class TestScatterAPI(unittest.TestCase):
    def setUp(self):
        self.places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))
        self.executed_api()

    def executed_api(self):
        self.scatter = paddle.scatter

    def check_static_result(self, place):
        with base.program_guard(base.Program(), base.Program()):
            input = paddle.static.data(
                name="input", shape=[3, 2], dtype="float64"
            )
            index = paddle.static.data(name="index", shape=[4], dtype="int64")
            updates = paddle.static.data(
                name="updates", shape=[4, 2], dtype="float64"
            )
            result = self.scatter(input, index, updates, False)

            input_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float64)
            index_data = np.array([2, 1, 0, 1]).astype(np.int64)
            updates_data = np.array([[1, 1], [2, 2], [3, 3], [4, 4]]).astype(
                np.float64
            )

            exe = base.Executor(place)
            fetches = exe.run(
                base.default_main_program(),
                feed={
                    "input": input_data,
                    "index": index_data,
                    "updates": updates_data,
                },
                fetch_list=[result],
            )
            self.assertEqual(
                (
                    fetches[0] == np.array([[3.0, 3.0], [6.0, 6.0], [1.0, 1.0]])
                ).all(),
                True,
            )

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                x_data = np.array([[1, 1], [2, 2], [3, 3]]).astype(np.float64)
                index_data = np.array([2, 1, 0, 1]).astype(np.int64)
                updates_data = np.array(
                    [[1, 1], [2, 2], [3, 3], [4, 4]]
                ).astype(np.float64)

                x = base.dygraph.to_variable(x_data)
                index = base.dygraph.to_variable(index_data)
                updates = base.dygraph.to_variable(updates_data)

                output1 = self.scatter(x, index, updates, overwrite=False)
                self.assertEqual(
                    (
                        output1.numpy()
                        == np.array([[3.0, 3.0], [6.0, 6.0], [1.0, 1.0]])
                    ).all(),
                    True,
                )

    def test_large_data(self):
        if os.name == "nt" or not paddle.is_compiled_with_cuda():
            return

        x = np.random.rand(183826, 256).astype("float32")
        index = np.ones(10759233, dtype="int64")
        updates = np.ones(shape=[10759233, 256], dtype="float32")

        def test_dygraph():
            with base.dygraph.guard():
                gpu_out = paddle.scatter(
                    paddle.to_tensor(x),
                    paddle.to_tensor(index),
                    paddle.to_tensor(updates),
                )
                return gpu_out.numpy()

        @switch_to_static_graph
        def test_static_graph():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                x_t = paddle.static.data(name="x", dtype=x.dtype, shape=x.shape)
                index_t = paddle.static.data(
                    name="index", dtype=index.dtype, shape=index.shape
                )
                updates_t = paddle.static.data(
                    name="updates", dtype=updates.dtype, shape=updates.shape
                )
                out_t = paddle.scatter(x_t, index_t, updates_t)
                feed = {
                    x_t.name: x,
                    index_t.name: index,
                    updates_t.name: updates,
                }
                fetch = [out_t]

                gpu_exe = paddle.static.Executor(paddle.CUDAPlace(0))
                gpu_value = gpu_exe.run(feed=feed, fetch_list=fetch)[0]
                return gpu_value

        np.testing.assert_array_equal(test_dygraph(), test_static_graph())


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestScatterOpFp16(OpTest):
    def setUp(self):
        self.__class__.op_type = "scatter"
        self.python_api = paddle.scatter
        # compute grad in the following code handly.
        self.__class__.no_need_check_grad = True
        self.x_type = 'float16'
        self.x_np = np.ones((3, 3)).astype(self.x_type)
        self.index_np = np.array([1, 2]).astype("int32")
        self.updates_np = np.random.random((2, 3)).astype(self.x_type)
        self.output_np = np.copy(self.x_np)
        self.output_np[self.index_np] = self.updates_np
        self.dout_np = np.random.random((3, 3)).astype(self.x_type)

        # compute ref_dx
        self.ref_dx = np.copy(self.dout_np)
        zero_np = np.zeros((2, 3)).astype(self.x_type)
        self.ref_dx[self.index_np] = zero_np

    def compute_ref_grad_updates(self):
        ref_grad_updates = paddle.gather(
            paddle.to_tensor(self.dout_np), paddle.to_tensor(self.index_np)
        )
        return ref_grad_updates

    def test_scatter_fp16(self):
        paddle.disable_static(place=paddle.CUDAPlace(0))
        x_tensor = paddle.to_tensor(self.x_np, stop_gradient=False)
        index_tensor = paddle.to_tensor(self.index_np)
        updates_tensor = paddle.to_tensor(self.updates_np, stop_gradient=False)
        out_tensor = paddle.scatter(x_tensor, index_tensor, updates_tensor)
        paddle.autograd.backward(
            [out_tensor], [paddle.to_tensor(self.dout_np)], retain_graph=True
        )
        ref_grad_updates = self.compute_ref_grad_updates()
        np.testing.assert_allclose(
            ref_grad_updates.numpy(False),
            updates_tensor.grad.numpy(False),
            rtol=1e-5,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            self.ref_dx, x_tensor.grad.numpy(False), rtol=1e-5, atol=1e-5
        )


class TestScatterInplaceAPI(TestScatterAPI):
    def executed_api(self):
        self.scatter = paddle.scatter_


@unittest.skipIf(core.is_compiled_with_cuda(), "CUDA will not throw exception")
class TestScatterError(unittest.TestCase):
    def test_scatter_index(self):
        paddle.disable_static()
        x = paddle.to_tensor([[1, 1], [2, 2], [3, 3]], dtype='float32')

        def test_neg_index():
            index = paddle.to_tensor([2, 1, -1, 1], dtype='int64')
            updates = paddle.to_tensor(
                [[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32'
            )
            out = paddle.scatter(x, index, updates)

        self.assertRaises(IndexError, test_neg_index)

        def test_too_big_index():
            index = paddle.to_tensor([2, 1, 5, 1], dtype='int64')
            updates = paddle.to_tensor(
                [[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32'
            )
            out = paddle.scatter(x, index, updates)

        self.assertRaises(IndexError, test_too_big_index)
        paddle.enable_static()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
