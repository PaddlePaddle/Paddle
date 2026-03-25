# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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
    convert_float_to_uint16,
    get_device_place,
    get_places,
    is_custom_device,
)

import paddle
from paddle import base
from paddle.base import core


def np_masked_scatter(x, mask, value):
    x, mask = np.broadcast_arrays(x, mask)
    mask_prefix_sum = np.clip(mask.cumsum() - 1, a_min=0, a_max=None)
    value = value.flatten()[mask_prefix_sum].reshape(x.shape)
    return np.where(mask, value, x)


paddle.enable_static()


class TestMaskedScatterError(unittest.TestCase):
    def setUp(self):
        self.init()

        self.x_np = np.random.random(self.x_shape).astype(self.dtype)
        self.mask_np = np.array(
            np.random.randint(2, size=self.mask_shape), dtype='bool'
        )

        self.value_np = np.random.randn(*self.value_shape).astype(self.dtype)

    def init(self):
        self.x_shape = (50, 3)
        self.mask_shape = self.x_shape
        self.dtype = "float32"
        self.value_shape = (300, 300)

    def test_mask_error(self):
        x = paddle.to_tensor(self.x_np, dtype=self.dtype)
        mask = paddle.to_tensor(self.mask_np).astype('int32')
        value = paddle.to_tensor(self.value_np, dtype=self.dtype)

        with np.testing.assert_raises(AssertionError):
            paddle.masked_scatter(x, mask, value)

    def test_dtype_error(self):
        x = paddle.to_tensor(self.x_np, dtype=self.dtype)
        mask = paddle.to_tensor(self.mask_np).astype('bool')
        value = paddle.to_tensor(self.value_np, dtype='float64')
        with np.testing.assert_raises(AssertionError):
            paddle.masked_scatter(x, mask, value)

    @unittest.skipIf(
        core.is_compiled_with_cuda(),
        "core is compiled with CUDA",
    )
    def test_numel_error(self):
        paddle.disable_static()
        self.value_np = np.random.randn(5, 5).astype(self.dtype)
        x = paddle.to_tensor(self.x_np, dtype=self.dtype)
        mask = paddle.to_tensor(self.mask_np).astype('bool')
        value = paddle.to_tensor(self.value_np, dtype=self.dtype)
        with np.testing.assert_raises(AssertionError):
            paddle.masked_scatter(x, mask, value)

    @unittest.skipIf(
        not core.is_compiled_with_cuda(),
        "core is not compiled with CUDA",
    )
    def test_numel_error_cuda(self):
        # The size check kernel uses asm("trap;") which fatally corrupts the
        # CUDA context.  Run in a subprocess so the parent stays healthy.
        code = """
import numpy as np
import paddle
paddle.disable_static()
x_np = np.random.random((50, 3)).astype("float32")
mask_np = np.ones((50, 3), dtype="bool")
value_np = np.random.randn(5, 5).astype("float32")
x = paddle.to_tensor(x_np)
mask = paddle.to_tensor(mask_np)
value = paddle.to_tensor(value_np)
out = paddle.masked_scatter(x, mask, value)
# Force synchronization so the device-side trap error surfaces.
paddle.device.cuda.synchronize()
"""
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        # Device-side printf may go to stdout; the OSError traceback is
        # on stderr.  Check both for the kernel error message.
        combined = (proc.stdout + proc.stderr).lower()
        self.assertTrue(
            "number of true elements in mask" in combined
            or "cuda error" in combined
            or "hip error" in combined
            or "device-side assert" in combined
            or "abort" in combined,
            f"Expected masked_scatter size-check error, got:\n{combined}",
        )


class TestMaskedScatterAPI(unittest.TestCase):
    def setUp(self):
        self.init()

        self.x_np = np.random.random(self.x_shape).astype(self.dtype)
        self.mask_np = np.array(
            np.random.randint(2, size=self.mask_shape), dtype="bool"
        )

        self.value_np = np.random.randn(*self.value_shape).astype(self.dtype)
        self.out_np = np_masked_scatter(self.x_np, self.mask_np, self.value_np)

    def init(self):
        self.x_shape = (50, 3)
        self.mask_shape = self.x_shape
        self.dtype = "float32"
        self.value_shape = (300, 300)

    def test_static_graph(self):
        paddle.enable_static()
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(startup_program, train_program):
            x = paddle.static.data(
                name='x', dtype=self.dtype, shape=self.x_shape
            )
            mask = paddle.static.data(
                name='mask', dtype='bool', shape=self.mask_shape
            )
            value = paddle.static.data(
                name='value', dtype=self.dtype, shape=self.value_np.shape
            )
            out = paddle.masked_scatter(x, mask, value)

            place = get_device_place()
            exe = base.Executor(place)
            res = exe.run(
                base.default_main_program(),
                feed={
                    'x': self.x_np,
                    'mask': self.mask_np,
                    'value': self.value_np,
                },
                fetch_list=[out],
            )
            np.testing.assert_allclose(
                res[0], self.out_np, atol=1e-5, rtol=1e-5
            )
            paddle.disable_static()

    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np, dtype=self.dtype)
        mask = paddle.to_tensor(self.mask_np).astype('bool')
        value = paddle.to_tensor(self.value_np, dtype=self.dtype)
        result = paddle.masked_scatter(x, mask, value)
        np.testing.assert_allclose(self.out_np, result.numpy(), rtol=1e-05)

        paddle.enable_static()


class TestMaskedScatterAPI1(TestMaskedScatterAPI):
    def init(self):
        self.x_shape = (6, 8, 9, 18)
        self.mask_shape = self.x_shape
        self.dtype = "float32"
        self.value_shape = (300, 300)


class TestMaskedScatterAPI2(TestMaskedScatterAPI):
    def init(self):
        self.x_shape = (168,)
        self.mask_shape = self.x_shape
        self.dtype = "float32"
        self.value_shape = (300, 300)


class TestMaskedScatterAPI3(TestMaskedScatterAPI):
    def init(self):
        self.x_shape = (6, 8, 9, 18)
        self.mask_shape = self.x_shape
        self.dtype = "float32"
        self.value_shape = (300, 300)


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA",
)
class TestMaskedScatterFP16API1(TestMaskedScatterAPI):
    def init(self):
        self.x_shape = (6, 8, 9, 18)
        self.mask_shape = self.x_shape
        self.dtype = "float16"
        self.value_shape = (300, 300)


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA",
)
class TestMaskedScatterFP16API2(TestMaskedScatterAPI):
    def init(self):
        self.x_shape = (168,)
        self.mask_shape = self.x_shape
        self.dtype = "float16"
        self.value_shape = (300, 300)


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA",
)
class TestMaskedScatterFP16API3(TestMaskedScatterAPI):
    def init(self):
        self.x_shape = (168,)
        self.mask_shape = self.x_shape
        self.dtype = "float16"
        self.value_shape = (300, 300)


class TestMaskedScatterAPIBroadcast(TestMaskedScatterAPI):
    def init(self):
        self.x_shape = (3, 40)
        self.mask_shape = (3, 1)
        self.dtype = "float32"
        self.value_shape = (300, 300)


class TestMaskedScatterAPIBroadcast2(TestMaskedScatterAPI):
    def init(self):
        self.x_shape = (3, 3)
        self.mask_shape = (1, 3)
        self.dtype = "float32"
        self.value_shape = (300, 300)


class TestMaskedScatterAPIBroadcast3(TestMaskedScatterAPI):
    def init(self):
        self.x_shape = (120,)
        self.mask_shape = (300, 120)
        self.dtype = "float32"
        self.value_shape = (300, 300)


class TestMaskedScatterAPIBroadcast4(TestMaskedScatterAPI):
    def init(self):
        self.x_shape = (300, 40)
        self.mask_shape = (40,)
        self.dtype = "float32"
        self.value_shape = (300, 300)


class TestMaskedScatterAPIBroadcast5(TestMaskedScatterAPI):
    def init(self):
        self.x_shape = (300, 40)
        self.mask_shape = (40,)
        self.dtype = "float32"
        self.value_shape = (300, 300)


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA",
)
class TestMaskedScatterFP16APIBroadcast(TestMaskedScatterAPI):
    def init(self):
        self.x_shape = (3, 40)
        self.mask_shape = (3, 1)
        self.dtype = "float16"
        self.value_shape = (300, 300)


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA",
)
class TestMaskedScatterFP16APIBroadcast2(TestMaskedScatterAPI):
    def init(self):
        self.x_shape = (300, 1)
        self.mask_shape = (300, 40)
        self.dtype = "float16"
        self.value_shape = (300, 300)


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA",
)
class TestMaskedScatterFP16APIBroadcast3(TestMaskedScatterAPI):
    def init(self):
        self.x_shape = (300, 1)
        self.mask_shape = (300, 40)
        self.dtype = "float16"
        self.value_shape = (300, 300)


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_bfloat16_supported(get_device_place()),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestMaskedScatterBF16(TestMaskedScatterAPI):
    def init(self):
        self.x_shape = (300, 1)
        self.mask_shape = (300, 1)
        self.dtype = "uint16"
        self.value_shape = (300, 300)

    def setUp(self):
        self.init()

        self.x_np = convert_float_to_uint16(
            np.random.random(self.x_shape).astype("float32")
        )
        self.mask_np = np.array(
            np.random.randint(2, size=self.mask_shape), dtype="bool"
        )

        self.value_np = convert_float_to_uint16(
            np.random.randn(*self.value_shape).astype("float32")
        )
        self.out_np = np_masked_scatter(self.x_np, self.mask_np, self.value_np)


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_bfloat16_supported(get_device_place()),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestMaskedScatterBF16APIBroadcast2(TestMaskedScatterBF16):
    def init(self):
        self.x_shape = (300, 1)
        self.mask_shape = (300, 3)
        self.dtype = "uint16"
        self.value_shape = (300, 300)


class TestMaskedScatterCPU(TestMaskedScatterAPI):
    """Explicitly run masked_scatter tests on CPUPlace to guarantee CPU
    coverage regardless of whether the build includes CUDA."""

    def test_static_graph(self):
        paddle.enable_static()
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(startup_program, train_program):
            x = paddle.static.data(
                name='x', dtype=self.dtype, shape=self.x_shape
            )
            mask = paddle.static.data(
                name='mask', dtype='bool', shape=self.mask_shape
            )
            value = paddle.static.data(
                name='value', dtype=self.dtype, shape=self.value_np.shape
            )
            out = paddle.masked_scatter(x, mask, value)

            place = core.CPUPlace()
            exe = base.Executor(place)
            res = exe.run(
                base.default_main_program(),
                feed={
                    'x': self.x_np,
                    'mask': self.mask_np,
                    'value': self.value_np,
                },
                fetch_list=[out],
            )
            np.testing.assert_allclose(
                res[0], self.out_np, atol=1e-5, rtol=1e-5
            )
            paddle.disable_static()

    def test_dygraph(self):
        paddle.disable_static(paddle.CPUPlace())
        x = paddle.to_tensor(self.x_np, dtype=self.dtype)
        mask = paddle.to_tensor(self.mask_np).astype('bool')
        value = paddle.to_tensor(self.value_np, dtype=self.dtype)
        result = paddle.masked_scatter(x, mask, value)
        np.testing.assert_allclose(self.out_np, result.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_dygraph_grad(self):
        paddle.disable_static(paddle.CPUPlace())
        x = paddle.to_tensor(self.x_np, dtype=self.dtype)
        x.stop_gradient = False
        mask = paddle.to_tensor(self.mask_np).astype('bool')
        value = paddle.to_tensor(self.value_np, dtype=self.dtype)
        value.stop_gradient = False
        result = paddle.masked_scatter(x, mask, value)
        loss = paddle.sum(result)
        loss.backward()
        self.assertEqual(list(x.grad.shape), list(self.x_np.shape))
        self.assertEqual(list(value.grad.shape), list(self.value_np.shape))
        paddle.enable_static()


class TestMaskedScatterCPU1(TestMaskedScatterCPU):
    def init(self):
        self.x_shape = (6, 8, 9, 18)
        self.mask_shape = self.x_shape
        self.dtype = "float32"
        self.value_shape = (300, 300)


class TestMaskedScatterCPU2(TestMaskedScatterCPU):
    def init(self):
        self.x_shape = (168,)
        self.mask_shape = self.x_shape
        self.dtype = "float32"
        self.value_shape = (300, 300)


class TestMaskedScatterCPUFloat64(TestMaskedScatterCPU):
    def init(self):
        self.x_shape = (50, 3)
        self.mask_shape = self.x_shape
        self.dtype = "float64"
        self.value_shape = (300, 300)


class TestMaskedScatterCPUBroadcast(TestMaskedScatterCPU):
    def init(self):
        self.x_shape = (3, 40)
        self.mask_shape = (3, 1)
        self.dtype = "float32"
        self.value_shape = (300, 300)


class TestMaskedScatterCPUBroadcast2(TestMaskedScatterCPU):
    def init(self):
        self.x_shape = (3, 3)
        self.mask_shape = (1, 3)
        self.dtype = "float32"
        self.value_shape = (300, 300)


class TestMaskedScatterCPUBroadcast3(TestMaskedScatterCPU):
    def init(self):
        self.x_shape = (120,)
        self.mask_shape = (300, 120)
        self.dtype = "float32"
        self.value_shape = (300, 300)


class TestMaskedScatterCPUBroadcast4(TestMaskedScatterCPU):
    def init(self):
        self.x_shape = (300, 40)
        self.mask_shape = (40,)
        self.dtype = "float32"
        self.value_shape = (300, 300)


class TestMaskedScatterAPI_ZeroSize(unittest.TestCase):
    def setUp(self):
        self.init()

        self.x_np = np.random.random(self.x_shape).astype(self.dtype)
        self.mask_np = np.array(
            np.random.randint(2, size=self.mask_shape), dtype="bool"
        )

        self.value_np = np.random.randn(*self.value_shape).astype(self.dtype)
        self.out_np = np_masked_scatter(self.x_np, self.mask_np, self.value_np)

        self.places = get_places()

    def init(self):
        self.x_shape = (3, 0)
        self.mask_shape = self.x_shape
        self.dtype = "float32"
        self.value_shape = (300, 300)

    def _test_dygraph(self, place):
        paddle.disable_static(place)
        x = paddle.to_tensor(self.x_np, dtype=self.dtype)
        x.stop_gradient = False
        mask = paddle.to_tensor(self.mask_np).astype('bool')
        value = paddle.to_tensor(self.value_np, dtype=self.dtype)
        result = paddle.masked_scatter(x, mask, value)
        np.testing.assert_allclose(self.out_np, result.numpy(), rtol=1e-05)
        paddle.sum(result).backward()
        np.testing.assert_allclose(x.grad.shape, x.shape)
        paddle.enable_static()

    def test_dygraph(self):
        for place in self.places:
            self._test_dygraph(place)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
