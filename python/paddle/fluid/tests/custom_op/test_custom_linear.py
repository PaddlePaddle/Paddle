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

import paddle
import paddle.static as static
import paddle.nn.functional as F
from paddle.utils.cpp_extension import load, get_build_directory
from paddle.utils.cpp_extension.extension_utils import run_cmd
from utils import paddle_includes, extra_cc_args, extra_nvcc_args
from paddle.fluid.framework import _test_eager_guard

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = '{}\\custom_linear\\custom_linear.pyd'.format(get_build_directory())
if os.name == 'nt' and os.path.isfile(file):
    cmd = 'del {}'.format(file)
    run_cmd(cmd, True)

custom_ops = load(
    name='custom_linear_jit',
    sources=['custom_linear_op.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cc flags
    extra_cuda_cflags=extra_nvcc_args,  # test for nvcc flags
    verbose=True)


def linear_dynamic(func, dtype, np_x, np_weight, np_bias):
    paddle.set_device("cpu")
    x = paddle.to_tensor(np_x, dtype=dtype)
    weight = paddle.to_tensor(np_weight, dtype=dtype)
    bias = paddle.to_tensor(np_bias, dtype=dtype)
    out = func(x, weight, bias)
    return out.numpy()


def linear_static(func, dtype, np_x, np_weight, np_bias):
    paddle.enable_static()
    paddle.set_device("cpu")
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name="x", shape=np_x.shape, dtype=dtype)
            weight = static.data(
                name="weight", shape=np_weight.shape, dtype=dtype)
            bias = static.data(name="bias", shape=np_bias.shape, dtype=dtype)
            out = func(x, weight, bias)

            exe = static.Executor()
            exe.run(static.default_startup_program())

            out_v, = exe.run(static.default_main_program(),
                             feed={
                                 "x": np_x.astype(dtype),
                                 "weight": np_weight.astype(dtype),
                                 "bias": np_bias.astype(dtype)
                             },
                             fetch_list=[out.name])
    paddle.disable_static()
    return out_v


class TestCustomLinearJit(unittest.TestCase):
    def setUp(self):
        self.dtypes = ['float32', 'float64']
        self.np_x = np.random.random((3, 2)).astype("float32")
        self.np_weight = np.full([2, 4], fill_value=0.5, dtype="float32")
        self.np_bias = np.ones([4], dtype="float32")

    def check_output(self, out, pd_out, name):
        self.assertTrue(
            np.array_equal(out, pd_out),
            "custom op {}: {},\n paddle api {}: {}".format(name, out, name,
                                                           pd_out))

    def test_static(self):
        for dtype in self.dtypes:
            pten_out = linear_static(custom_ops.pten_linear, dtype, self.np_x,
                                     self.np_weight, self.np_bias)
            pd_out = linear_static(F.linear, dtype, self.np_x, self.np_weight,
                                   self.np_bias)
            self.check_output(pten_out, pd_out, "pten_out")

    def func_dynamic(self):
        for dtype in self.dtypes:
            pten_out = linear_dynamic(custom_ops.pten_linear, dtype, self.np_x,
                                      self.np_weight, self.np_bias)
            pd_out = linear_dynamic(F.linear, dtype, self.np_x, self.np_weight,
                                    self.np_bias)
            self.check_output(pten_out, pd_out, "pten_out")

    def test_dynamic(self):
        with _test_eager_guard():
            self.func_dynamic()
        self.func_dynamic()


if __name__ == "__main__":
    unittest.main()
