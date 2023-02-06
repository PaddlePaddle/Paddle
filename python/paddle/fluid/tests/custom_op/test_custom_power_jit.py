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

import os
import unittest

import numpy as np
from utils import extra_cc_args, paddle_includes

import paddle
import paddle.static as static
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = '{}\\custom_power_jit\\custom_power_jit.pyd'.format(
    get_build_directory()
)
if os.name == 'nt' and os.path.isfile(file):
    cmd = 'del {}'.format(file)
    run_cmd(cmd, True)

custom_module = load(
    name='custom_power_jit',
    sources=['custom_power.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cc flags
    verbose=True,
)


def custom_power_dynamic(func, device, dtype, np_x, use_func=True):
    paddle.set_device(device)

    t = paddle.to_tensor(np_x, dtype=dtype)
    t.stop_gradient = False

    out = func(t) if use_func else paddle.pow(t, 2)
    out.stop_gradient = False

    out.backward()
    if t.grad is None:
        return out.numpy(), t.grad
    else:
        return out.numpy(), t.grad.numpy()


def custom_power_static(func, device, dtype, np_x, use_func=True):
    paddle.enable_static()
    paddle.set_device(device)

    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name='X', shape=[None, 8], dtype=dtype)
            x.stop_gradient = False
            out = func(x) if use_func else paddle.pow(x, 2)
            static.append_backward(out)

            exe = static.Executor()
            exe.run(static.default_startup_program())
            # in static graph mode, x data has been covered by out
            out_v = exe.run(
                static.default_main_program(),
                feed={'X': np_x},
                fetch_list=[out.name],
            )

    paddle.disable_static()
    return out_v


class TestJITLoad(unittest.TestCase):
    def setUp(self):
        self.custom_op = custom_module.custom_power
        self.devices = ['cpu']
        self.dtypes = ['float32', 'float64']
        if paddle.is_compiled_with_cuda():
            self.devices.append('gpu')
            self.dtypes.append('float16')

    def test_static(self):
        for device in self.devices:
            for dtype in self.dtypes:
                if device == 'cpu' and dtype == 'float16':
                    continue
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)

                out = custom_power_static(self.custom_op, device, dtype, x)
                pd_out = custom_power_static(
                    self.custom_op, device, dtype, x, False
                )
                np.testing.assert_allclose(out, pd_out, rtol=1e-5, atol=1e-8)

    def test_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                if device == 'cpu' and dtype == 'float16':
                    continue
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)

                out, x_grad = custom_power_dynamic(
                    self.custom_op, device, dtype, x
                )
                pd_out, pd_x_grad = custom_power_dynamic(
                    self.custom_op, device, dtype, x, False
                )
                np.testing.assert_allclose(out, pd_out, rtol=1e-5, atol=1e-8)
                np.testing.assert_allclose(
                    x_grad, pd_x_grad, rtol=1e-5, atol=1e-8
                )


if __name__ == '__main__':
    unittest.main()
