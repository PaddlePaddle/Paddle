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
from utils import extra_cc_args, extra_nvcc_args, paddle_includes

import paddle
import paddle.static as static
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd
=======
import numpy as np

import paddle
import paddle.static as static
from paddle.utils.cpp_extension import load, get_build_directory
from paddle.utils.cpp_extension.extension_utils import run_cmd
from utils import paddle_includes, extra_cc_args, extra_nvcc_args
from paddle.fluid.framework import _test_eager_guard
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = '{}\\custom_conj\\custom_conj.pyd'.format(get_build_directory())
if os.name == 'nt' and os.path.isfile(file):
    cmd = 'del {}'.format(file)
    run_cmd(cmd, True)

custom_ops = load(
    name='custom_conj_jit',
    sources=['custom_conj_op.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cc flags
    extra_cuda_cflags=extra_nvcc_args,  # test for nvcc flags
<<<<<<< HEAD
    verbose=True,
)


def is_complex(dtype):
    return (
        dtype == paddle.fluid.core.VarDesc.VarType.COMPLEX64
        or dtype == paddle.fluid.core.VarDesc.VarType.COMPLEX128
    )
=======
    verbose=True)


def is_complex(dtype):
    return dtype == paddle.fluid.core.VarDesc.VarType.COMPLEX64 or \
      dtype == paddle.fluid.core.VarDesc.VarType.COMPLEX128
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def to_complex(dtype):
    if dtype == "float32":
        return np.complex64
    elif dtype == "float64":
        return np.complex128
    else:
        return dtype


def conj_dynamic(func, dtype, np_input):
    paddle.set_device("cpu")
    x = paddle.to_tensor(np_input)
    out = func(x)
    out.stop_gradient = False
    sum_out = paddle.sum(out)
    if is_complex(sum_out.dtype):
        sum_out.real().backward()
    else:
        sum_out.backward()
    if x.grad is None:
        return out.numpy(), x.grad
    else:
        return out.numpy(), x.grad.numpy()


def conj_static(func, shape, dtype, np_input):
    paddle.enable_static()
    paddle.set_device("cpu")
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name="x", shape=shape, dtype=dtype)
            x.stop_gradient = False
            out = func(x)
            sum_out = paddle.sum(out)
            static.append_backward(sum_out)

            exe = static.Executor()
            exe.run(static.default_startup_program())

<<<<<<< HEAD
            out_v, x_grad_v = exe.run(
                static.default_main_program(),
                feed={"x": np_input},
                fetch_list=[out.name, x.name + "@GRAD"],
            )
=======
            out_v, x_grad_v = exe.run(static.default_main_program(),
                                      feed={"x": np_input},
                                      fetch_list=[out.name, x.name + "@GRAD"])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    paddle.disable_static()
    return out_v, x_grad_v


class TestCustomConjJit(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.dtypes = ['float32', 'float64']
        self.shape = [2, 20, 2, 3]

    def check_output(self, out, pd_out, name):
        np.testing.assert_array_equal(
            out,
            pd_out,
            err_msg='custom op {}: {},\n paddle api {}: {}'.format(
<<<<<<< HEAD
                name, out, name, pd_out
            ),
        )
=======
                name, out, name, pd_out))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def run_dynamic(self, dtype, np_input):
        out, x_grad = conj_dynamic(custom_ops.custom_conj, dtype, np_input)
        pd_out, pd_x_grad = conj_dynamic(paddle.conj, dtype, np_input)

        self.check_output(out, pd_out, "out")
        self.check_output(x_grad, pd_x_grad, "x's grad")

    def run_static(self, dtype, np_input):
<<<<<<< HEAD
        out, x_grad = conj_static(
            custom_ops.custom_conj, self.shape, dtype, np_input
        )
        pd_out, pd_x_grad = conj_static(
            paddle.conj, self.shape, dtype, np_input
        )
=======
        out, x_grad = conj_static(custom_ops.custom_conj, self.shape, dtype,
                                  np_input)
        pd_out, pd_x_grad = conj_static(paddle.conj, self.shape, dtype,
                                        np_input)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.check_output(out, pd_out, "out")
        self.check_output(x_grad, pd_x_grad, "x's grad")

<<<<<<< HEAD
    def test_dynamic(self):
=======
    def func_dynamic(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        for dtype in self.dtypes:
            np_input = np.random.random(self.shape).astype(dtype)
            self.run_dynamic(dtype, np_input)

<<<<<<< HEAD
=======
    def test_dynamic(self):
        with _test_eager_guard():
            self.func_dynamic()
        self.func_dynamic()

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_static(self):
        for dtype in self.dtypes:
            np_input = np.random.random(self.shape).astype(dtype)
            self.run_static(dtype, np_input)

    # complex only used in dynamic mode now
    def test_complex_dynamic(self):
        for dtype in self.dtypes:
            np_input = np.random.random(self.shape).astype(
<<<<<<< HEAD
                dtype
            ) + 1j * np.random.random(self.shape).astype(dtype)
=======
                dtype) + 1j * np.random.random(self.shape).astype(dtype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.run_dynamic(to_complex(dtype), np_input)


if __name__ == "__main__":
    unittest.main()
