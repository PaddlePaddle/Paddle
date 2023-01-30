# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
<<<<<<< HEAD

import numpy as np
from utils import extra_cc_args, paddle_includes

import paddle
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd

=======
import paddle
import numpy as np
from paddle.utils.cpp_extension import load, get_build_directory
from utils import paddle_includes, extra_cc_args
from paddle.utils.cpp_extension.extension_utils import run_cmd
from paddle.fluid.framework import _test_eager_guard
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = '{}\\dispatch_op\\dispatch_op.pyd'.format(get_build_directory())
if os.name == 'nt' and os.path.isfile(file):
    cmd = 'del {}'.format(file)
    run_cmd(cmd, True)

dispatch_op = load(
    name='dispatch_op',
    sources=['dispatch_test_op.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,
<<<<<<< HEAD
    verbose=True,
)


class TestJitDispatch(unittest.TestCase):
    def setUp(self):
        paddle.set_device('cpu')

    def run_dispatch_test(self, func, dtype):
=======
    verbose=True)


class TestJitDispatch(unittest.TestCase):

    def setUp(self):
        paddle.set_device('cpu')

    def run_dispatch_test_impl(self, func, dtype):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        np_x = np.ones([2, 2]).astype(dtype)
        x = paddle.to_tensor(np_x)
        out = func(x)
        np_x = x.numpy()
        np_out = out.numpy()
        self.assertTrue(dtype in str(np_out.dtype))
        np.testing.assert_array_equal(
            np_x,
            np_out,
<<<<<<< HEAD
            err_msg='custom op x: {},\n custom op out: {}'.format(np_x, np_out),
        )
=======
            err_msg='custom op x: {},\n custom op out: {}'.format(np_x, np_out))

    def run_dispatch_test(self, func, dtype):
        with _test_eager_guard():
            self.run_dispatch_test_impl(func, dtype)
        self.run_dispatch_test_impl(func, dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_dispatch_integer(self):
        dtypes = ["int32", "int64", "int8", "uint8", "int16"]
        for dtype in dtypes:
            self.run_dispatch_test(dispatch_op.dispatch_test_integer, dtype)

    def test_dispatch_complex(self):
        dtypes = ["complex64", "complex128"]
        for dtype in dtypes:
            self.run_dispatch_test(dispatch_op.dispatch_test_complex, dtype)

    def test_dispatch_float_and_integer(self):
        dtypes = [
<<<<<<< HEAD
            "float32",
            "float64",
            "int32",
            "int64",
            "int8",
            "uint8",
            "int16",
        ]
        for dtype in dtypes:
            self.run_dispatch_test(
                dispatch_op.dispatch_test_float_and_integer, dtype
            )
=======
            "float32", "float64", "int32", "int64", "int8", "uint8", "int16"
        ]
        for dtype in dtypes:
            self.run_dispatch_test(dispatch_op.dispatch_test_float_and_integer,
                                   dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_dispatch_float_and_complex(self):
        dtypes = ["float32", "float64", "complex64", "complex128"]
        for dtype in dtypes:
<<<<<<< HEAD
            self.run_dispatch_test(
                dispatch_op.dispatch_test_float_and_complex, dtype
            )

    def test_dispatch_float_and_integer_and_complex(self):
        dtypes = [
            "float32",
            "float64",
            "int32",
            "int64",
            "int8",
            "uint8",
            "int16",
            "complex64",
            "complex128",
        ]
        for dtype in dtypes:
            self.run_dispatch_test(
                dispatch_op.dispatch_test_float_and_integer_and_complex, dtype
            )
=======
            self.run_dispatch_test(dispatch_op.dispatch_test_float_and_complex,
                                   dtype)

    def test_dispatch_float_and_integer_and_complex(self):
        dtypes = [
            "float32", "float64", "int32", "int64", "int8", "uint8", "int16",
            "complex64", "complex128"
        ]
        for dtype in dtypes:
            self.run_dispatch_test(
                dispatch_op.dispatch_test_float_and_integer_and_complex, dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_dispatch_float_and_half(self):
        dtypes = ["float32", "float64", "float16"]
        for dtype in dtypes:
<<<<<<< HEAD
            self.run_dispatch_test(
                dispatch_op.dispatch_test_float_and_half, dtype
            )
=======
            self.run_dispatch_test(dispatch_op.dispatch_test_float_and_half,
                                   dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


if __name__ == '__main__':
    unittest.main()
