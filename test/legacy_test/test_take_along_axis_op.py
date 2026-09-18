#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import subprocess
import sys
import unittest

import numpy as np
from op_test import (
    OpTest,
    convert_float_to_uint16,
    get_device_place,
    get_places,
    is_custom_device,
)
from utils import dygraph_guard

import paddle
from paddle.framework import core

paddle.enable_static()


class TestTakeAlongAxis0Size(OpTest):
    def setUp(self):
        self.python_api = paddle.take_along_axis
        self.op_type = "take_along_axis"
        self.dtype = "float64"
        self.check_pir = True

        x = np.zeros((2, 0, 5)).astype(self.dtype)
        indices = np.zeros((2, 3, 5)).astype("int64")

        self.inputs = {'Input': x, 'Index': indices}
        self.attrs = {'Axis': 1}

        output = np.zeros((2, 3, 5)).astype(self.dtype)
        self.outputs = {'Result': output}

    def test_check_output(self):
        self.check_output(check_pir=self.check_pir)

    def test_check_grad(self):
        self.check_grad(['Input'], 'Result', check_pir=self.check_pir)


class TestTakeAlongAxis0Size2(OpTest):
    def setUp(self):
        self.python_api = paddle.take_along_axis
        self.op_type = "take_along_axis"
        self.dtype = "float64"
        self.check_pir = True

        x = np.random.rand(2, 3, 5).astype(self.dtype)
        indices = np.zeros((2, 0, 5)).astype("int64")

        self.inputs = {'Input': x, 'Index': indices}
        self.attrs = {'Axis': 1}

        output = np.zeros((2, 0, 5)).astype(self.dtype)
        self.outputs = {'Result': output}

    def test_check_output(self):
        self.check_output(check_pir=self.check_pir)

    def test_check_grad(self):
        self.grad = np.zeros_like(self.outputs['Result']).astype(self.dtype)
        self.check_grad(
            ['Input'],
            'Result',
            user_defined_grads=[self.grad],
            check_pir=self.check_pir,
        )


class TestTakeAlongAxisOp(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "take_along_axis"
        self.prim_op_type = "prim"
        self.python_api = paddle.tensor.take_along_axis
        self.public_python_api = paddle.tensor.take_along_axis
        self.check_cinn = True
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.target = np.take_along_axis(self.xnp, self.index, self.axis)
        broadcast_shape_list = list(self.x_shape)
        broadcast_shape_list[self.axis] = 1
        self.broadcast_shape = tuple(broadcast_shape_list)
        self.index_broadcast = np.broadcast_to(self.index, self.broadcast_shape)
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index_broadcast,
        }
        self.attrs = {'Axis': self.axis}
        self.outputs = {'Result': self.target}

    def test_check_output(self):
        self.check_output(check_cinn=self.check_cinn, check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['Input'],
            'Result',
            check_cinn=self.check_cinn,
            check_pir=True,
            check_prim_pir=True,
        )

    def init_data(self):
        self.x_type = "float64"
        self.x_shape = (5, 5, 5)
        self.index_type = "int32"
        self.axis = 2
        dim_size = self.x_shape[self.axis]
        self.index = np.random.randint(
            -dim_size, dim_size, size=(5, 1, 1)
        ).astype(self.index_type)
        self.axis_type = "int64"


class TestTakeAlongAxisDuplicatedIndices(TestTakeAlongAxisOp):
    def init_data(self):
        self.dtype = np.float32
        self.x_type = "float32"
        self.x_shape = (5, 6, 7)
        self.index_type = "int64"
        self.axis = 2
        dim_size = self.x_shape[self.axis]
        self.index = (
            np.asarray([-dim_size, -dim_size, dim_size - 1, dim_size - 1, 0])
            .astype(self.index_type)
            .reshape([5, 1, 1])
        )
        self.axis_type = "int64"

    def test_check_output(self):
        self.check_output(
            check_cinn=self.check_cinn, check_pir=True, check_prim_pir=True
        )

    def test_check_grad(self):
        self.check_grad(
            ['Input'],
            'Result',
            check_cinn=self.check_cinn,
            check_pir=True,
            check_prim_pir=True,
        )


class TestTakeAlongAxisFP16Op(TestTakeAlongAxisOp):
    def init_data(self):
        self.dtype = np.float16
        self.x_type = "float16"
        self.x_shape = (5, 5, 5)
        self.index_type = "int32"
        self.axis = 2
        dim_size = self.x_shape[self.axis]
        self.index = np.random.randint(
            -dim_size, dim_size, size=(5, 1, 1)
        ).astype(self.index_type)
        self.axis_type = "int64"


class TestTakeAlongAxisOp2(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "take_along_axis"
        self.python_api = paddle.tensor.take_along_axis
        self.check_cinn = True
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.target = np.zeros((2, 3, 4)).astype(self.x_type)
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    self.target[i, j, k] = self.xnp[i, j, self.index[i, j, k]]
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index,
        }
        self.attrs = {'Axis': self.axis, 'broadcast': False}
        self.outputs = {'Result': self.target}

    def init_data(self):
        self.x_type = "float64"
        self.x_shape = (10, 10, 10)
        self.index_type = "int64"
        self.axis = 2
        dim_size = self.x_shape[self.axis]
        self.index = np.random.randint(-dim_size, dim_size, (2, 3, 4)).astype(
            self.index_type
        )
        self.axis_type = "int64"


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_bfloat16_supported(get_device_place()),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestTakeAlongAxisBF16Op(OpTest):
    def setUp(self):
        self.init_data()
        self.op_type = "take_along_axis"
        self.prim_op_type = "prim"
        self.python_api = paddle.tensor.take_along_axis
        self.public_python_api = paddle.tensor.take_along_axis
        self.check_cinn = True
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.target = np.take_along_axis(self.xnp, self.index, self.axis)
        broadcast_shape_list = list(self.x_shape)
        broadcast_shape_list[self.axis] = 1
        self.broadcast_shape = tuple(broadcast_shape_list)
        self.index_broadcast = np.broadcast_to(self.index, self.broadcast_shape)
        self.inputs = {
            'Input': self.xnp,
            'Index': self.index_broadcast,
        }
        self.attrs = {'Axis': self.axis}
        self.outputs = {'Result': self.target}

        self.inputs['Input'] = convert_float_to_uint16(self.inputs['Input'])
        self.outputs['Result'] = convert_float_to_uint16(self.outputs['Result'])
        self.place = get_device_place()

    def test_check_output(self):
        self.check_output_with_place(
            self.place, check_cinn=self.check_cinn, check_pir=True
        )

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ['Input'],
            'Result',
            check_cinn=self.check_cinn,
            check_pir=True,
            check_prim_pir=True,
        )

    def init_data(self):
        self.dtype = np.uint16
        self.x_type = "float32"
        self.x_shape = (5, 5, 5)
        self.index_type = "int32"
        self.axis = 2
        dim_size = self.x_shape[self.axis]
        self.index = np.random.randint(
            -dim_size, dim_size, size=(5, 1, 1)
        ).astype(self.index_type)
        self.axis_type = "int64"


class TestCase1(TestTakeAlongAxisOp):
    def init_data(self):
        self.x_type = "float64"
        self.x_shape = (5, 5, 5)
        self.index_type = "int32"
        self.axis = 0
        dim_size = self.x_shape[self.axis]
        self.index = np.random.randint(
            -dim_size, dim_size, size=(1, 1, 5)
        ).astype(self.index_type)
        self.axis_type = "int64"


@unittest.skipIf(
    not core.is_compiled_with_cuda() or core.is_compiled_with_rocm(),
    "deterministic path only runs on CUDA (compiled out on ROCm/DCU, where "
    "RadixSortPairs is unavailable and backward falls back to atomic scatter)",
)
class TestTakeAlongAxisGradDeterministic(TestTakeAlongAxisDuplicatedIndices):
    """Exercises TakeAlongAxisGradDeterministicKernel (FLAGS_cudnn_deterministic).

    test_check_grad  – gradient values must match the numeric reference.
    test_deterministic_grad – two identical backward passes must produce
                              bit-identical results.
    """

    def init_data(self):
        self.dtype = np.float64
        self.x_type = "float64"
        self.x_shape = (5, 6, 7)
        self.index_type = "int64"
        self.axis = 2
        self.broadcast = True
        # Author the index at full x shape (no implicit broadcasting in the
        # harness). Values 0 (x4) and 1 (x3) repeat along axis=2, so several
        # out_grad elements accumulate into the same x_grad cell.
        self.index = np.broadcast_to(
            np.asarray([0, 0, 1, 1, 1, 0, 0]).reshape(1, 1, 7), self.x_shape
        ).astype(self.index_type)
        self.axis_type = "int64"

    def setUp(self):
        self.init_data()
        self.op_type = "take_along_axis"
        self.prim_op_type = "prim"
        self.python_api = paddle.tensor.take_along_axis
        self.public_python_api = paddle.tensor.take_along_axis
        self.check_cinn = True
        self.xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.index_broadcast = self.index.astype(self.index_type)
        # broadcast=False makes index smaller than x on the non-axis dims;
        # slicing x down to the index extents lets np.take_along_axis build the
        # reference for both modes (a no-op when the shapes already match).
        slices = tuple(
            slice(0, self.index_broadcast.shape[d])
            if d != self.axis
            else slice(None)
            for d in range(len(self.x_shape))
        )
        self.inputs = {'Input': self.xnp, 'Index': self.index_broadcast}
        self.attrs = {'Axis': self.axis, 'broadcast': self.broadcast}
        self.outputs = {
            'Result': np.take_along_axis(
                self.xnp[slices], self.index_broadcast, self.axis
            )
        }

    # Deterministic path is a property of the composite backward kernel; keep
    # prim/CINN (compiler-optimized) paths out of these tests so the fixed
    # reduction order under test is never routed through a fusing backend.
    # def test_check_output(self):
    #     self.check_output(check_pir=True)

    def test_check_grad(self):
        paddle.set_flags({'FLAGS_cudnn_deterministic': True})
        try:
            self.check_grad(['Input'], 'Result', check_pir=True)
        finally:
            paddle.set_flags({'FLAGS_cudnn_deterministic': False})

    def test_deterministic_grad(self):
        paddle.disable_static()
        paddle.set_flags({'FLAGS_cudnn_deterministic': True})
        try:
            x = paddle.to_tensor(
                self.xnp, place=paddle.CUDAPlace(0), stop_gradient=False
            )
            idx = paddle.to_tensor(
                self.index_broadcast, place=paddle.CUDAPlace(0)
            )

            out1 = paddle.take_along_axis(
                x, idx, self.axis, broadcast=self.broadcast
            )
            out1.sum().backward()
            grad1 = x.grad.numpy().copy()
            x.clear_grad()

            out2 = paddle.take_along_axis(
                x, idx, self.axis, broadcast=self.broadcast
            )
            out2.sum().backward()
            grad2 = x.grad.numpy().copy()

            np.testing.assert_array_equal(
                grad1,
                grad2,
                err_msg="Deterministic grad produced different results "
                "across two identical backward passes.",
            )
        finally:
            paddle.set_flags({'FLAGS_cudnn_deterministic': False})
            paddle.enable_static()


class TestTakeAlongAxisGradDeterministicInt32Index(
    TestTakeAlongAxisGradDeterministic
):
    """int32 indices must run normally on the deterministic backward path."""

    def init_data(self):
        super().init_data()
        self.index_type = "int32"


class TestTakeAlongAxisGradDeterministicBroadcastFalse(
    TestTakeAlongAxisGradDeterministic
):
    """
    broadcast=False: index is smaller than x in the non-axis dims.
    Representative case: x=(10,10,10), index=(2,3,4), axis=2.
    """

    def init_data(self):
        self.dtype = np.float64
        self.x_type = "float64"
        self.x_shape = (10, 10, 10)
        self.index_type = "int64"
        self.axis = 2
        self.broadcast = False
        dim_size = self.x_shape[self.axis]
        self.index = np.random.randint(0, dim_size, (2, 3, 4)).astype(
            self.index_type
        )
        self.axis_type = "int64"


class TestTakeAlongAxisGradDeterministicNegativeIndex(
    TestTakeAlongAxisGradDeterministic
):
    """Valid negative indices must be normalized before the sorted reduction.

    -7 -> 0 (x4) and -6 -> 1 (x3): the same effective mapping as the parent's
    positive index, authored as negatives to exercise index normalization while
    keeping the fixed accumulation order verifiable via check_grad.
    """

    def init_data(self):
        super().init_data()
        self.index = np.broadcast_to(
            np.asarray([-7, -7, -6, -6, -6, -7, -7]).reshape(1, 1, 7),
            self.x_shape,
        ).astype(self.index_type)


@unittest.skipIf(
    not core.is_compiled_with_cuda() or core.is_compiled_with_rocm(),
    "deterministic path only runs on CUDA (compiled out on ROCm/DCU, where "
    "RadixSortPairs is unavailable and backward falls back to atomic scatter)",
)
class TestTakeAlongAxisGradDeterministicDtypes(unittest.TestCase):
    """Deterministic backward for every supported floating dtype.

    For each dtype it checks (1) the gradient matches an independent fp64
    reference within the dtype tolerance and (2) two identical runs are
    bit-identical, i.e. the accumulation order is fixed. A valid negative index
    (-8 -> 0, -5 -> 3) is included so normalization is covered here too.
    """

    def _run(self, np_name, rtol, atol):
        np.random.seed(2024)
        x_shape = (4, 8, 5)
        axis = 1
        axis_size = x_shape[axis]
        idx_line = np.asarray([0, 0, -8, 3, 3, 3, -5, 2])  # duplicates + neg
        idx_np = np.broadcast_to(
            idx_line.reshape(1, axis_size, 1), x_shape
        ).astype("int64")
        x_np = np.random.randn(*x_shape)
        gout_np = np.random.randn(*x_shape) * 10.0

        # Independent fp64 reference: scatter-add the upstream grad by the
        # normalized index along the axis.
        idx_norm = idx_np.copy()
        idx_norm[idx_norm < 0] += axis_size
        g_ref = np.zeros(x_shape, dtype="float64")
        for i in range(x_shape[0]):
            for k in range(axis_size):
                for j in range(x_shape[2]):
                    g_ref[i, idx_norm[i, k, j], j] += gout_np[i, k, j]

        paddle.disable_static()
        paddle.set_flags({'FLAGS_cudnn_deterministic': True})
        try:
            place = paddle.CUDAPlace(0)
            paddle_dtype = getattr(paddle, np_name)

            def run_once():
                x = paddle.to_tensor(x_np, place=place).cast(paddle_dtype)
                x.stop_gradient = False
                idx = paddle.to_tensor(idx_np, place=place)
                out = paddle.take_along_axis(x, idx, axis)
                gout = paddle.to_tensor(gout_np, place=place).cast(paddle_dtype)
                paddle.autograd.backward([out], [gout])
                return x.grad.cast("float64").numpy()

            g1 = run_once()
            g2 = run_once()
            # (2) fixed accumulation order -> bit-identical across runs.
            np.testing.assert_array_equal(g1, g2)
            # (1) correct within dtype tolerance vs the fp64 reference.
            np.testing.assert_allclose(g1, g_ref, rtol=rtol, atol=atol)
        finally:
            paddle.set_flags({'FLAGS_cudnn_deterministic': False})
            paddle.enable_static()

    def test_float32(self):
        self._run("float32", rtol=1e-5, atol=1e-5)

    def test_float16(self):
        self._run("float16", rtol=1e-2, atol=1e-1)

    def test_bfloat16(self):
        if not core.is_bfloat16_supported(paddle.CUDAPlace(0)):
            self.skipTest("bfloat16 is not supported on this device")
        self._run("bfloat16", rtol=3e-2, atol=3e-1)


@unittest.skipIf(
    not core.is_compiled_with_cuda() or core.is_compiled_with_rocm(),
    "deterministic path only runs on CUDA (compiled out on ROCm/DCU, where "
    "RadixSortPairs is unavailable and backward falls back to atomic scatter)",
)
class TestTakeAlongAxisGradDeterministicIndexOutOfBounds(unittest.TestCase):
    """Out-of-range indices must be rejected by the deterministic backward path."""

    def _run_out_of_bounds(self, index_value):
        # axis=2 has size 4, so both index_value == 4 (positive overflow) and
        # index_value == -5 (negative overflow, still negative after += 4) are
        # out of the valid [-4, 4) range and must trip the bounds check.
        #
        # The grad op is invoked directly instead of through backward(): a full
        # forward+backward would trap in the forward kernel first and never
        # reach grad kernel, which is the code path under test here.
        code = f"""
import numpy as np
import paddle
paddle.disable_static()
paddle.set_device("gpu")
paddle.set_flags({{'FLAGS_cudnn_deterministic': True}})
arr = paddle.to_tensor(
    np.random.random((2, 3, 4)).astype("float32"), place=paddle.CUDAPlace(0)
)
idx = paddle.full((2, 3, 4), {index_value}, dtype="int64")
out_grad = paddle.ones((2, 3, 4), dtype="float32")
paddle._C_ops.take_along_axis_grad(arr, idx, out_grad, 2)
# Force synchronization so the device-side trap error surfaces.
paddle.device.cuda.synchronize()
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        # Device-side printf may go to stdout; the traceback lands on stderr.
        combined = (proc.stdout + proc.stderr).lower()
        self.assertTrue(
            "index is out of bounds" in combined
            or "cuda error" in combined
            or "hip error" in combined
            or "device-side assert" in combined
            or "trap" in combined
            or "abort" in combined,
            f"Expected an out-of-bounds index error, got:\n{combined}",
        )

    def test_positive_out_of_bounds(self):
        self._run_out_of_bounds(4)

    def test_negative_out_of_bounds(self):
        self._run_out_of_bounds(-5)


@unittest.skipIf(
    not core.is_compiled_with_cuda() or core.is_compiled_with_rocm(),
    "deterministic path only runs on CUDA (compiled out on ROCm/DCU, where "
    "RadixSortPairs is unavailable and backward falls back to atomic scatter)",
)
class TestTakeAlongAxisGradDeterministicIllegalIndexDtype(unittest.TestCase):
    """Unsupported index dtype must be rejected on the deterministic backward.

    A full forward+backward would be rejected by the forward gather kernel
    first, which shares the same int32/int64 dtype check, and would never
    reach the backward kernel under test.
    """

    def setUp(self):
        paddle.disable_static()
        paddle.set_device("gpu")
        paddle.set_flags({'FLAGS_cudnn_deterministic': True})
        self.arr = paddle.to_tensor(
            np.random.random((2, 3, 4)).astype("float32"),
            place=paddle.CUDAPlace(0),
        )
        self.out_grad = paddle.ones((2, 3, 4), dtype="float32")

    def tearDown(self):
        paddle.set_flags({'FLAGS_cudnn_deterministic': False})
        paddle.enable_static()

    def test_int16_index_rejected(self):
        idx = paddle.full((2, 3, 4), 1, dtype="int16")
        with self.assertRaises(ValueError):
            paddle._C_ops.take_along_axis_grad(self.arr, idx, self.out_grad, 2)
            paddle.device.cuda.synchronize()


class TestTakeAlongAxisAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [3, 3]
        self.index_shape = [1, 3]
        self.axis = 0
        dim_size = self.shape[self.axis]
        self.index_np = np.random.randint(
            -dim_size, dim_size, size=([1, 3])
        ).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = get_places()

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape)
            index = paddle.static.data('Index', self.index_shape, "int64")
            out = paddle.take_along_axis(x, index, self.axis)
            exe = paddle.static.Executor(self.place[0])
            res = exe.run(
                feed={'X': self.x_np, 'Index': self.index_np}, fetch_list=[out]
            )
        out_ref = np.array(
            np.take_along_axis(self.x_np, self.index_np, self.axis)
        )
        for out in res:
            np.testing.assert_allclose(out, out_ref, rtol=0.001)

    def test_api_dygraph(self):
        paddle.disable_static(self.place[0])
        x_tensor = paddle.to_tensor(self.x_np)
        self.index = paddle.to_tensor(self.index_np)
        out = paddle.take_along_axis(x_tensor, self.index, self.axis)
        out_ref = np.array(
            np.take_along_axis(self.x_np, self.index_np, self.axis)
        )
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)
        paddle.enable_static()

    def test_api_dygraph_dtype(self):
        if sys.platform == 'darwin' or sys.platform == 'win32':
            return
        paddle.disable_static(paddle.CPUPlace())
        with self.assertRaises(AssertionError):
            x_tensor = paddle.to_tensor(self.x_np)
            self.index = paddle.to_tensor(self.index_np).astype("float32")
            out = paddle.take_along_axis(x_tensor, self.index, self.axis)
            out_ref = np.array(
                np.take_along_axis(self.x_np, self.index_np, self.axis)
            )
            np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)
        paddle.enable_static()


class TestTakeAlongAxisAPICase1(TestTakeAlongAxisAPI):
    def setUp(self):
        np.random.seed(0)
        self.shape = [2, 2]
        self.index_shape = [4, 2]
        self.axis = 0
        dim_size = self.shape[self.axis]
        self.index_np = np.random.randint(
            -dim_size, dim_size, size=(4, 2)
        ).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = get_places()


class TestTakeAlongAxisAPICase2(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [3, 3]
        self.index_shape = [1, 3]
        self.axis = 0
        dim_size = self.shape[self.axis]
        self.index_np = np.random.randint(
            -dim_size, dim_size, size=(1, 3)
        ).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = get_places()

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.shape)
            index = paddle.static.data('Index', self.index_shape, "int64")
            out = paddle.take_along_axis(x, index, self.axis, False)
            exe = paddle.static.Executor(self.place[0])
            res = exe.run(
                feed={'X': self.x_np, 'Index': self.index_np}, fetch_list=[out]
            )
        out_ref = np.zeros_like(self.index_np, dtype=self.x_np.dtype)
        for i in range(self.index_shape[0]):
            for j in range(self.index_shape[1]):
                out_ref[i, j] = self.x_np[self.index_np[i, j], j]
        for out in res:
            np.testing.assert_allclose(out, out_ref, rtol=0.001)

    def test_api_dygraph(self):
        paddle.disable_static(self.place[0])
        x_tensor = paddle.to_tensor(self.x_np)
        self.index = paddle.to_tensor(self.index_np)
        out = paddle.take_along_axis(x_tensor, self.index, self.axis, False)
        out_ref = np.zeros_like(self.index_np, dtype=self.x_np.dtype)
        for i in range(self.index_shape[0]):
            for j in range(self.index_shape[1]):
                out_ref[i, j] = self.x_np[self.index_np[i, j], j]
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=0.001)
        paddle.enable_static()

    def test_error(self):
        paddle.disable_static(self.place[0])
        tensorx = paddle.to_tensor([[1, 2, 3], [4, 5, 6]]).astype("float32")
        indices = paddle.to_tensor([1]).astype("int32")
        # len(arr.shape) != len(indices.shape)
        with self.assertRaises(ValueError):
            res = paddle.take_along_axis(tensorx, indices, 0, False)
        # the element of indices out of range
        # (only catch cpu assertion though gpu can raise exception)
        with self.assertRaises(IndexError):
            indices = paddle.to_tensor([[100]]).astype("int32")
            res = paddle.take_along_axis(
                tensorx.to("cpu"), indices.to("cpu"), 0, False
            )
        with self.assertRaises(IndexError):
            indices = paddle.to_tensor([[-100]]).astype("int32")
            res = paddle.take_along_axis(
                tensorx.to("cpu"), indices.to("cpu"), 0, False
            )
        # the shape of indices doesn't match
        with self.assertRaises(RuntimeError):
            indices = paddle.to_tensor(
                [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
            ).astype("int32")
            res = paddle.take_along_axis(tensorx, indices, 0, False)


class TestTakeAlongAxisAPICase4(unittest.TestCase):
    def test_static_shape_take_along_axis(self):
        with dygraph_guard():
            x = paddle.randn([4, 2])
            ind = paddle.to_tensor([[0, 1]])

            static_f = paddle.jit.to_static(
                paddle.take_along_axis,
                input_spec=[
                    paddle.static.InputSpec(
                        shape=[-1, -1], dtype="float32", name="arr"
                    ),
                    paddle.static.InputSpec(
                        shape=[-1, 2], dtype="int64", name="indices"
                    ),
                ],
                full_graph=True,
            )
            _ = static_f(x, ind, axis=0, broadcast=False)


class TestTakeAlongAxis_ZeroSize(OpTest):
    def setUp(self):
        self.python_api = paddle.take_along_axis
        self.op_type = "take_along_axis"
        self.dtype = "float64"
        self.check_pir = True

        x = np.zeros((2, 0, 5)).astype(self.dtype)
        indices = np.zeros((2, 3, 5)).astype("int64")

        self.inputs = {'Input': x, 'Index': indices}
        self.attrs = {'Axis': 1}

        output = np.zeros((2, 3, 5)).astype(self.dtype)
        self.outputs = {'Result': output}

    def test_check_output(self):
        self.check_output_with_place(
            paddle.CPUPlace(), check_pir=self.check_pir
        )
        if core.is_compiled_with_cuda() or is_custom_device():
            self.check_output_with_place(
                get_device_place(), check_pir=self.check_pir
            )

    def test_check_grad(self):
        self.check_grad_with_place(
            paddle.CPUPlace(), ['Input'], 'Result', check_pir=self.check_pir
        )
        if core.is_compiled_with_cuda() or is_custom_device():
            self.check_grad_with_place(
                get_device_place(),
                ['Input'],
                'Result',
                check_pir=self.check_pir,
            )


class TestTakeAlongAxisInt16(TestTakeAlongAxisOp):
    def init_data(self):
        self.dtype = np.int16
        self.x_type = "int16"
        self.x_shape = (5, 5, 5)
        self.index_type = "int32"
        self.axis = 2
        dim_size = self.x_shape[self.axis]
        self.index = np.random.randint(
            -dim_size, dim_size, size=(5, 1, 1)
        ).astype(self.index_type)
        self.axis_type = "int64"

    def test_check_grad(self):
        """int16 does not require and allow for grad check"""
        pass


class TestTakeAlongAxisUInt8(TestTakeAlongAxisOp):
    def init_data(self):
        self.dtype = np.uint8
        self.x_type = "uint8"
        self.x_shape = (5, 5, 5)
        self.index_type = "int32"
        self.axis = 2
        dim_size = self.x_shape[self.axis]
        self.index = np.random.randint(
            -dim_size, dim_size, size=(5, 1, 1)
        ).astype(self.index_type)
        self.axis_type = "int64"

    def test_check_grad(self):
        """uint8 does not require and allow for grad check"""
        pass


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
