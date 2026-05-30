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

import hashlib
import math
import os
import platform
import sys
import unittest

import numpy as np
from op import Operator
from op_test import (
    OpTest,
    convert_uint16_to_float,
    get_device,
    get_device_place,
    get_places,
    is_custom_device,
)
from utils import dygraph_guard

import paddle
from paddle import base
from paddle.base import Program, core, program_guard
from paddle.base.framework import convert_np_dtype_to_dtype_
from paddle.tensor import random


def _float_bits(value, dtype):
    value = np.asarray(value, dtype=dtype)
    if value.dtype == np.float32:
        return int(value.view(np.uint32))
    if value.dtype == np.float64:
        return int(value.view(np.uint64))
    return None


def _chunked_mean(out, chunk_size):
    flat = out.ravel()
    total = np.float64(0.0)
    for start in range(0, flat.size, chunk_size):
        chunk = flat[start : start + chunk_size]
        total = np.float64(total + np.sum(chunk, dtype=np.float64))
    return np.float64(total / flat.size)


def _debug_reference_mean(out, expect_mean):
    flat = out.ravel()
    fsum_mean = np.float64(math.fsum(map(float, flat)) / flat.size)
    longdouble_mean = np.sum(out, dtype=np.longdouble) / np.longdouble(
        flat.size
    )
    longdouble_mean_f64 = np.float64(longdouble_mean)
    chunked_mean_1m = _chunked_mean(out, 1_000_000)
    chunked_mean_16k = _chunked_mean(out, 16_384)
    print(
        "[uniform_random_debug] "
        f"reference_fsum_mean={fsum_mean!r} "
        f"diff={fsum_mean - expect_mean!r} "
        f"bits={_float_bits(fsum_mean, np.float64)}"
    )
    print(
        "[uniform_random_debug] "
        f"reference_longdouble_mean={longdouble_mean!r} "
        f"as_float64={longdouble_mean_f64!r} "
        f"diff={longdouble_mean_f64 - expect_mean!r} "
        f"bits={_float_bits(longdouble_mean_f64, np.float64)}"
    )
    print(
        "[uniform_random_debug] "
        f"reference_chunked_mean_1m={chunked_mean_1m!r} "
        f"diff={chunked_mean_1m - expect_mean!r} "
        f"bits={_float_bits(chunked_mean_1m, np.float64)}"
    )
    print(
        "[uniform_random_debug] "
        f"reference_chunked_mean_16k={chunked_mean_16k!r} "
        f"diff={chunked_mean_16k - expect_mean!r} "
        f"bits={_float_bits(chunked_mean_16k, np.float64)}"
    )


def _debug_uniform_random_output(
    case_name, out, expect_mean, expect_std, sample, expect
):
    mean = np.mean(out)
    std = np.std(out)
    expect_mean = np.asarray(expect_mean, dtype=np.asarray(mean).dtype)
    expect_std = np.asarray(expect_std, dtype=np.asarray(std).dtype)
    expect_sample = np.asarray(expect, dtype=sample.dtype)
    out_bytes = np.ascontiguousarray(out).view(np.uint8)

    print(f"[uniform_random_debug] case={case_name}")
    print(
        "[uniform_random_debug] "
        f"python={sys.version.split()[0]} "
        f"platform={platform.platform()} "
        f"numpy={np.__version__} "
        f"paddle={paddle.__version__}"
    )
    print(
        "[uniform_random_debug] "
        f"numpy_version={np.__version__} numpy_file={np.__file__}"
    )
    try:
        print(
            "[uniform_random_debug] "
            f"device={paddle.device.cuda.get_device_name()} "
            f"cuda={paddle.version.cuda()} "
            f"cudnn={paddle.version.cudnn()}"
        )
    except Exception as exc:
        print(f"[uniform_random_debug] device_info_error={exc!r}")
    try:
        flags = paddle.get_flags(
            ["FLAGS_deterministic_rng", "FLAGS_deterministic_rng_grid"]
        )
        print(f"[uniform_random_debug] paddle_flags={flags}")
    except Exception as exc:
        print(f"[uniform_random_debug] paddle_flags_error={exc!r}")
    print(
        "[uniform_random_debug] "
        f"env_OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
        f"env_OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')} "
        f"env_MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}"
    )
    print(
        "[uniform_random_debug] "
        f"shape={out.shape} dtype={out.dtype} strides={out.strides} "
        f"c_contiguous={out.flags.c_contiguous}"
    )
    print(
        "[uniform_random_debug] "
        f"sha256={hashlib.sha256(out_bytes).hexdigest()} "
        f"min={np.min(out)!r} max={np.max(out)!r}"
    )
    print(
        "[uniform_random_debug] "
        f"mean={mean!r} expect_mean={expect_mean!r} "
        f"mean_diff={mean - expect_mean!r} "
        f"mean_bits={_float_bits(mean, np.asarray(mean).dtype)} "
        f"expect_mean_bits={_float_bits(expect_mean, expect_mean.dtype)}"
    )
    _debug_reference_mean(out, expect_mean)
    print(
        "[uniform_random_debug] "
        f"std={std!r} expect_std={expect_std!r} "
        f"std_diff={std - expect_std!r} "
        f"std_bits={_float_bits(std, np.asarray(std).dtype)} "
        f"expect_std_bits={_float_bits(expect_std, expect_std.dtype)}"
    )
    print(
        "[uniform_random_debug] "
        "sample="
        f"{np.array2string(sample, precision=17, floatmode='unique')}"
    )
    print(
        "[uniform_random_debug] "
        "sample_expect="
        f"{np.array2string(expect_sample, precision=17, floatmode='unique')} "
        f"sample_max_abs_diff={np.max(np.abs(sample - expect_sample))!r}"
    )
    if hasattr(np, "show_runtime"):
        print("[uniform_random_debug] numpy_show_runtime_begin")
        np.show_runtime()
        print("[uniform_random_debug] numpy_show_runtime_end")
    else:
        print("[uniform_random_debug] numpy_config_begin")
        np.__config__.show()
        print("[uniform_random_debug] numpy_config_end")


def output_hist(out):
    if out.dtype == np.uint16:
        out = convert_uint16_to_float(out)
    hist, _ = np.histogram(out, range=(-5, 10))
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.1 * np.ones(10)
    return hist, prob


def output_hist_diag(out):
    diag_num = min(out.shape)
    for i in range(diag_num):
        assert abs(out[i][i] - 1.0) < 1e-9
        # ignore diagonal elements
        out[i][i] = 100
    hist, _ = np.histogram(out, range=(-5, 10))
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.1 * np.ones(10)
    return hist, prob


class TestUniformRandomOp_attr_tensorlist(OpTest):
    def setUp(self):
        self.op_type = "uniform_random"
        self.python_api = paddle.uniform
        self.new_shape = (1000, 784)
        shape_tensor = []
        for index, ele in enumerate(self.new_shape):
            shape_tensor.append(
                ("x" + str(index), np.ones(1).astype("int64") * ele)
            )
        self.inputs = {'ShapeTensorList': shape_tensor}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((1000, 784)).astype("float32")}

    def init_attrs(self):
        self.attrs = {"min": -5.0, "max": 10.0, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output, check_pir=True)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)


class TestMaxMinAreInt(TestUniformRandomOp_attr_tensorlist):
    def init_attrs(self):
        self.attrs = {"min": -5, "max": 10, "seed": 10}
        self.output_hist = output_hist


class TestUniformRandomOp_attr_tensorlist_int32(OpTest):
    def setUp(self):
        self.op_type = "uniform_random"
        self.python_api = paddle.uniform
        self.new_shape = (1000, 784)
        shape_tensor = []
        for index, ele in enumerate(self.new_shape):
            shape_tensor.append(
                ("x" + str(index), np.ones(1).astype("int32") * ele)
            )
        self.inputs = {'ShapeTensorList': shape_tensor}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((1000, 784)).astype("float32")}

    def init_attrs(self):
        self.attrs = {"min": -5.0, "max": 10.0, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output, check_pir=True)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)


class TestUniformRandomOp_attr_tensor(OpTest):
    def setUp(self):
        self.op_type = "uniform_random"
        self.python_api = paddle.uniform
        self.inputs = {"ShapeTensor": np.array([1000, 784]).astype("int64")}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((1000, 784)).astype("float32")}

    def init_attrs(self):
        self.attrs = {"min": -5.0, "max": 10.0, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output, check_pir=True)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)


class TestUniformRandomOp_attr_tensor_int32(OpTest):
    def setUp(self):
        self.op_type = "uniform_random"
        self.python_api = paddle.uniform
        self.inputs = {"ShapeTensor": np.array([1000, 784]).astype("int32")}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((1000, 784)).astype("float32")}

    def init_attrs(self):
        self.attrs = {"min": -5.0, "max": 10.0, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output, check_pir=True)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)


class TestUniformRandomOp(OpTest):
    def setUp(self):
        self.op_type = "uniform_random"
        self.python_api = paddle.uniform
        self.inputs = {}
        self.init_dtype()
        self.init_attrs()
        self.outputs = {"Out": np.zeros((1000, 784)).astype("float32")}

    def init_dtype(self):
        self.dtype = np.float32

    def init_attrs(self):
        self.attrs = {
            "shape": [1000, 784],
            "min": -5.0,
            "max": 10.0,
            "dtype": convert_np_dtype_to_dtype_(self.dtype),
        }
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output, check_pir=True)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)

    def test_check_api(self):
        places = self._get_places()
        for place in places:
            with base.dygraph.base.guard(place=place):
                out = self.python_api(
                    self.attrs['shape'],
                    self.dtype,
                    self.attrs['min'],
                    self.attrs['max'],
                )


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA",
)
class TestUniformRandomFP16Op(TestUniformRandomOp):
    def init_dtype(self):
        self.dtype = np.float16


class TestUniformRandomBF16Op(TestUniformRandomOp):
    def init_dtype(self):
        self.dtype = np.uint16


class TestUniformRandomComplex64Op(TestUniformRandomOp):
    def init_dtype(self):
        self.dtype = np.complex64

    def test_on_cpu(self):
        with dygraph_guard(), paddle.device.device_guard("cpu"):
            x = paddle.uniform([3, 3], paddle.complex64, -2, 2)


class TestUniformRandomComplex128Op(TestUniformRandomOp):
    def init_dtype(self):
        self.dtype = np.complex128

    def test_on_cpu(self):
        with dygraph_guard(), paddle.device.device_guard("cpu"):
            x = paddle.uniform([3, 3], paddle.complex128, -2, 2)


class TestUniformRandomOpError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        main_prog = Program()
        start_prog = Program()
        with program_guard(main_prog, start_prog):

            def test_Variable():
                x1 = base.create_lod_tensor(
                    np.zeros((4, 784)), [[1, 1, 1, 1]], base.CPUPlace()
                )
                paddle.uniform(x1)

            self.assertRaises(TypeError, test_Variable)

            def test_Variable2():
                x1 = np.zeros((4, 784))
                paddle.uniform(x1)

            self.assertRaises(TypeError, test_Variable2)

            def test_out_dtype():
                out = paddle.tensor.random.uniform(
                    shape=[3, 4], dtype='float64'
                )
                if paddle.framework.in_pir_mode():
                    self.assertEqual(out.dtype, base.core.DataType.FLOAT64)
                else:
                    self.assertEqual(out.dtype, paddle.float64)

            test_out_dtype()
        paddle.disable_static()


class TestUniformRandomOpWithDiagInit(TestUniformRandomOp):
    def init_attrs(self):
        self.attrs = {
            "shape": [1000, 784],
            "min": -5.0,
            "max": 10.0,
            "seed": 10,
            "diag_num": 784,
            "diag_step": 784,
            "diag_val": 1.0,
        }
        self.output_hist = output_hist_diag

    def test_check_output(self):
        self.check_output_customized(self.verify_output, check_pir=False)


class TestUniformRandomOpSelectedRows(unittest.TestCase):
    def test_check_output(self):
        for place in get_places():
            self.check_with_place(place)

    def check_with_place(self, place):
        scope = core.Scope()
        out = scope.var("X").get_selected_rows()
        paddle.seed(10)
        op = Operator(
            "uniform_random",
            Out="X",
            shape=[1000, 784],
            min=-5.0,
            max=10.0,
            seed=10,
        )
        op.run(scope, place)
        self.assertEqual(out.get_tensor().shape(), [1000, 784])
        hist, prob = output_hist(np.array(out.get_tensor()))
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)


class TestUniformRandomOpSelectedRowsWithDiagInit(
    TestUniformRandomOpSelectedRows
):
    def check_with_place(self, place):
        scope = core.Scope()
        out = scope.var("X").get_selected_rows()
        paddle.seed(10)
        op = Operator(
            "uniform_random",
            Out="X",
            shape=[500, 784],
            min=-5.0,
            max=10.0,
            seed=10,
            diag_num=500,
            diag_step=784,
            diag_val=1.0,
        )
        op.run(scope, place)
        self.assertEqual(out.get_tensor().shape(), [500, 784])
        hist, prob = output_hist_diag(np.array(out.get_tensor()))
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)


class TestUniformRandomOpApi(unittest.TestCase):
    def test_api(self):
        paddle.enable_static()
        paddle.seed(10)

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data('x', shape=[-1, 16], dtype='float32')

            linear = paddle.nn.Linear(
                in_features=x.shape[-1],
                out_features=16,
                weight_attr=paddle.nn.initializer.UniformInitializer(
                    low=-0.5,
                    high=0.5,
                    seed=10,
                    diag_num=16,
                    diag_step=16,
                    diag_val=1.0,
                ),
            )
            y = linear(x)

            place = base.CPUPlace()
            x_data = np.random.rand(3, 16).astype("float32")
            exe = base.Executor(place)
            exe.run(paddle.static.default_startup_program())
            ret = exe.run(
                paddle.static.default_main_program(),
                feed={'x': x_data},
                fetch_list=[y],
                return_numpy=False,
            )
        paddle.disable_static()


class TestUniformRandomOp_attr_tensor_API(unittest.TestCase):
    def test_attr_tensor_API(self):
        paddle.enable_static()
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(train_program, startup_program):
            dim_tensor = paddle.tensor.fill_constant([1], "int64", 3)
            ret = paddle.uniform([1, dim_tensor, 2])

            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = get_device_place()
            exe = base.Executor(place)

            exe.run(startup_program)
            outs = exe.run(train_program, fetch_list=[ret])
        paddle.disable_static()

    def test_attr_tensorlist_int32_API(self):
        paddle.enable_static()
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(train_program, startup_program):
            dim_1 = paddle.tensor.fill_constant([1], "int64", 3)
            dim_2 = paddle.tensor.fill_constant([1], "int32", 2)
            ret = paddle.uniform([1, dim_1, dim_2])

            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = get_device_place()
            exe = base.Executor(place)

            exe.run(startup_program)
            outs = exe.run(train_program, fetch_list=[ret])
        paddle.disable_static()

    def test_attr_tensor_int32_API(self):
        paddle.enable_static()
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(train_program, startup_program):
            shape = paddle.static.data(
                name='shape_tensor', shape=[2], dtype="int32"
            )
            ret = paddle.uniform(shape)

            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = get_device_place()
            exe = base.Executor(place)
            Shape = np.array([2, 3]).astype('int32')
            exe.run(startup_program)
            outs = exe.run(
                train_program, feed={'shape_tensor': Shape}, fetch_list=[ret]
            )
        paddle.disable_static()


class TestUniformRandomOp_API_seed(unittest.TestCase):
    def test_attr_tensor_API(self):
        paddle.enable_static()
        _seed = 10
        gen = paddle.seed(_seed)
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(train_program, startup_program):
            _min = 5
            _max = 10

            ret = paddle.uniform([2, 3, 2], min=_min, max=_max, seed=_seed)
            ret_2 = paddle.uniform([2, 3, 2], min=_min, max=_max, seed=_seed)
            res = paddle.equal(ret, ret_2)
            place = base.CPUPlace()
            if base.core.is_compiled_with_cuda():
                place = get_device_place()
            exe = base.Executor(place)

            exe.run(startup_program)
            ret_value, cmp_value = exe.run(train_program, fetch_list=[ret, res])
            self.assertTrue(np.array(cmp_value).all())
            for i in ret_value.flatten():
                self.assertGreaterEqual(i, _min)
                self.assertLess(i, _max)
        paddle.disable_static()


class TestUniformRandomOpSelectedRowsShapeTensor(unittest.TestCase):
    def test_check_output(self):
        for place in get_places():
            self.check_with_place(place)

    def check_with_place(self, place):
        scope = core.Scope()
        out = scope.var("X").get_selected_rows()
        shape_tensor = scope.var("Shape").get_tensor()
        shape_tensor.set(np.array([1000, 784]).astype("int64"), place)
        paddle.seed(10)
        op = Operator(
            "uniform_random",
            ShapeTensor="Shape",
            Out="X",
            min=-5.0,
            max=10.0,
            seed=10,
        )
        op.run(scope, place)
        self.assertEqual(out.get_tensor().shape(), [1000, 784])
        hist, prob = output_hist(np.array(out.get_tensor()))
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)


class TestUniformRandomOpSelectedRowsShapeTensorList(unittest.TestCase):
    def test_check_output(self):
        for place in get_places():
            self.check_with_place(place)

    def check_with_place(self, place):
        scope = core.Scope()
        out = scope.var("X").get_selected_rows()
        shape_1 = scope.var("shape1").get_tensor()
        shape_1.set(np.array([1000]).astype("int64"), place)
        shape_2 = scope.var("shape2").get_tensor()
        shape_2.set(np.array([784]).astype("int64"), place)
        paddle.seed(10)
        op = Operator(
            "uniform_random",
            ShapeTensorList=["shape1", "shape2"],
            Out="X",
            min=-5.0,
            max=10.0,
            seed=10,
        )
        op.run(scope, place)
        self.assertEqual(out.get_tensor().shape(), [1000, 784])
        hist, prob = output_hist(np.array(out.get_tensor()))
        np.testing.assert_allclose(hist, prob, rtol=0, atol=0.01)


class TestUniformRandomDygraphMode(unittest.TestCase):
    def test_check_output(self):
        with base.dygraph.guard():
            x = paddle.uniform([10], dtype="float32", min=0.0, max=1.0)
            x_np = x.numpy()
            for i in range(10):
                self.assertTrue(x_np[i] > 0 and x_np[i] < 1.0)


class TestUniformRandomBatchSizeLikeOpError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        main_prog = Program()
        start_prog = Program()
        with program_guard(main_prog, start_prog):

            def test_Variable():
                x1 = base.create_lod_tensor(
                    np.zeros((100, 784)), [[10, 10, 10, 70]], base.CPUPlace()
                )
                random.uniform_random_batch_size_like(x1)

            self.assertRaises(TypeError, test_Variable)

            def test_shape():
                x1 = paddle.static.data(
                    name='x2', shape=[-1, 100, 784], dtype='float32'
                )
                random.uniform_random_batch_size_like(x1, shape="shape")

            self.assertRaises(TypeError, test_shape)

            def test_dtype():
                x2 = paddle.static.data(
                    name='x2', shape=[-1, 100, 784], dtype='float32'
                )
                random.uniform_random_batch_size_like(x2, 'int32')

            self.assertRaises(TypeError, test_dtype)
        paddle.disable_static()


class TestUniformAlias(unittest.TestCase):
    def test_alias(self):
        paddle.uniform([2, 3], min=-5.0, max=5.0)
        paddle.tensor.uniform([2, 3], min=-5.0, max=5.0)
        paddle.tensor.random.uniform([2, 3], min=-5.0, max=5.0)

        def test_uniform_random():
            paddle.tensor.random.uniform_random([2, 3], min=-5.0, max=5.0)

        self.assertRaises(AttributeError, test_uniform_random)


class TestUniformOpError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        main_prog = Program()
        start_prog = Program()
        with program_guard(main_prog, start_prog):

            def test_Variable():
                x1 = base.create_lod_tensor(
                    np.zeros((100, 784)), [[10, 10, 10, 70]], base.CPUPlace()
                )
                paddle.tensor.random.uniform(x1)

            self.assertRaises(TypeError, test_Variable)

            def test_Variable2():
                x1 = np.zeros((100, 784))
                paddle.tensor.random.uniform(x1)

            self.assertRaises(TypeError, test_Variable2)

            def test_dtype():
                x2 = paddle.static.data(
                    name='x2', shape=[-1, 100, 784], dtype='float32'
                )
                paddle.tensor.random.uniform(x2, 'int32')

            self.assertRaises(TypeError, test_dtype)

            def test_out_dtype():
                out = paddle.tensor.random.uniform(
                    shape=[3, 4], dtype='float64'
                )
                if paddle.framework.in_pir_mode():
                    self.assertEqual(out.dtype, base.core.DataType.FLOAT64)
                else:
                    self.assertEqual(out.dtype, paddle.float64)

            test_out_dtype()
        paddle.disable_static()


class TestUniformDygraphMode(unittest.TestCase):
    def test_check_output(self):
        with base.dygraph.guard():
            x = paddle.tensor.random.uniform(
                [10], dtype="float32", min=0.0, max=1.0
            )
            x_np = x.numpy()
            for i in range(10):
                self.assertTrue(x_np[i] > 0 and x_np[i] < 1.0)


class TestUniformDtype(unittest.TestCase):
    def test_default_dtype(self):
        paddle.disable_static()

        def test_default_fp16():
            paddle.framework.set_default_dtype('float16')
            out = paddle.tensor.random.uniform([2, 3])
            self.assertEqual(out.dtype, paddle.float16)

        def test_default_fp32():
            paddle.framework.set_default_dtype('float32')
            out = paddle.tensor.random.uniform([2, 3])
            self.assertEqual(out.dtype, paddle.float32)

        def test_default_fp64():
            paddle.framework.set_default_dtype('float64')
            out = paddle.tensor.random.uniform([2, 3])
            self.assertEqual(out.dtype, paddle.float64)

        def test_dygraph_fp16():
            if not (paddle.is_compiled_with_cuda() or is_custom_device()):
                paddle.enable_static()
                return
            paddle.set_device(get_device())
            out = paddle.uniform([2, 3], dtype=paddle.float16)
            self.assertEqual(out.dtype, paddle.float16)

        if paddle.is_compiled_with_cuda() or is_custom_device():
            paddle.set_device(get_device())
            test_default_fp16()
        test_default_fp64()
        test_default_fp32()
        test_dygraph_fp16()

        paddle.enable_static()


class TestRandomValue(unittest.TestCase):
    def test_fixed_random_number(self):
        # Test GPU Fixed random number, which is generated by 'curandStatePhilox4_32_10_t'
        if not paddle.is_compiled_with_cuda():
            return

        # Different GPU generate different random value. Only test V100 here.
        if "V100" not in paddle.device.cuda.get_device_name():
            return

        print("Test Fixed Random number on V100 GPU------>")
        paddle.disable_static()

        paddle.set_device(get_device())
        paddle.seed(2021)

        expect_mean = 0.50000454338820143895816272561205551028251647949218750
        expect_std = 0.28867379167297479991560749112977646291255950927734375
        expect = [
            0.55298901,
            0.65184678,
            0.49375412,
            0.57943639,
            0.16459608,
            0.67181056,
            0.03021481,
            0.0238559,
            0.07742096,
            0.55972187,
        ]
        out = paddle.rand([32, 3, 1024, 1024], dtype='float64').numpy()
        _debug_uniform_random_output(
            "rand_float64",
            out,
            expect_mean,
            expect_std,
            out[2, 1, 512, 1000:1010],
            expect,
        )
        self.assertEqual(np.mean(out), expect_mean)
        self.assertEqual(np.std(out), expect_std)
        np.testing.assert_allclose(
            out[2, 1, 512, 1000:1010], expect, rtol=1e-05
        )

        expect_mean = 0.50002604722976684570312500
        expect_std = 0.2886914908885955810546875
        expect = [
            0.45320973,
            0.17582087,
            0.725341,
            0.30849215,
            0.622257,
            0.46352342,
            0.97228295,
            0.12771158,
            0.286525,
            0.9810645,
        ]
        out = paddle.rand([32, 3, 1024, 1024], dtype='float32').numpy()
        _debug_uniform_random_output(
            "rand_float32",
            out,
            expect_mean,
            expect_std,
            out[2, 1, 512, 1000:1010],
            expect,
        )
        self.assertEqual(np.mean(out), expect_mean)
        self.assertEqual(np.std(out), expect_std)
        np.testing.assert_allclose(
            out[2, 1, 512, 1000:1010], expect, rtol=1e-05
        )

        expect_mean = 25.11843109130859375
        expect_std = 43.370647430419921875
        expect = [
            30.089634,
            77.05225,
            3.1201615,
            68.34072,
            59.266724,
            -25.33281,
            12.973292,
            27.41127,
            -17.412298,
            27.931019,
        ]
        out = (
            paddle.empty([16, 16, 16, 16], dtype='float32')
            .uniform_(-50, 100)
            .numpy()
        )
        _debug_uniform_random_output(
            "uniform_inplace_float32",
            out,
            expect_mean,
            expect_std,
            out[10, 10, 10, 0:10],
            expect,
        )
        self.assertEqual(np.mean(out), expect_mean)
        self.assertEqual(np.std(out), expect_std)
        np.testing.assert_allclose(out[10, 10, 10, 0:10], expect, rtol=1e-05)

        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
