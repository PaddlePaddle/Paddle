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

import numpy as np
from utils import check_output, extra_cc_args, extra_nvcc_args, paddle_includes

import paddle
from paddle import static
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = f'{get_build_directory()}\\custom_conj\\custom_conj.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)

custom_ops = load(
    name='custom_conj_jit',
    sources=['custom_conj_op.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cc flags
    extra_cuda_cflags=extra_nvcc_args,  # test for nvcc flags
    verbose=True,
)


def is_complex(dtype):
    return dtype == paddle.complex64 or dtype == paddle.complex128


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

            if paddle.framework.in_pir_mode():
                ops = static.default_main_program().global_block().ops
                fetch_list = [out, ops[-1].result(0)]
            else:
                fetch_list = [out.name, x.name + "@GRAD"]

            out_v, x_grad_v = exe.run(
                static.default_main_program(),
                feed={"x": np_input},
                fetch_list=fetch_list,
            )
    paddle.disable_static()
    return out_v, x_grad_v


class TestCustomConjJit(unittest.TestCase):
    def setUp(self):
        self.dtypes = ['float32', 'float64']
        self.shape = [2, 20, 2, 3]

    def test_dynamic(self):
        for dtype in self.dtypes:
            np_input = np.random.random(self.shape).astype(dtype)
            out, x_grad = conj_dynamic(custom_ops.custom_conj, dtype, np_input)
            pd_out, pd_x_grad = conj_dynamic(paddle.conj, dtype, np_input)

            check_output(out, pd_out, "out")
            check_output(x_grad, pd_x_grad, "x's grad")

    def test_static(self):
        for dtype in self.dtypes:
            np_input = np.random.random(self.shape).astype(dtype)
            out, x_grad = conj_static(
                custom_ops.custom_conj, self.shape, dtype, np_input
            )
            pd_out, pd_x_grad = conj_static(
                paddle.conj, self.shape, dtype, np_input
            )

            check_output(out, pd_out, "out")
            check_output(x_grad, pd_x_grad, "x's grad")

    # complex only used in dynamic mode now
    def test_complex_dynamic(self):
        for dtype in self.dtypes:
            np_input = np.random.random(self.shape).astype(
                dtype
            ) + 1j * np.random.random(self.shape).astype(dtype)

            out, x_grad = conj_dynamic(
                custom_ops.custom_conj, to_complex(dtype), np_input
            )
            pd_out, pd_x_grad = conj_dynamic(
                paddle.conj, to_complex(dtype), np_input
            )

            check_output(out, pd_out, "out")
            check_output(x_grad, pd_x_grad, "x's grad")


if __name__ == "__main__":
    unittest.main()
