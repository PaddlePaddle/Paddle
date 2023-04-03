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
from utils import extra_cc_args, paddle_includes

import paddle
from paddle import static
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = f'{get_build_directory()}\\multi_out_jit\\multi_out_jit.pyd'
if os.name == 'nt' and os.path.isfile(file):
    cmd = f'del {file}'
    run_cmd(cmd, True)

# Compile and load custom op Just-In-Time.
multi_out_module = load(
    name='multi_out_jit',
    sources=['multi_out_test_op.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cflags
    verbose=True,
)


def discrete_out_dynamic(use_phi, device, dtype, np_w, np_x, np_y, np_z):
    paddle.set_device(device)
    w = paddle.to_tensor(np_w, dtype=dtype, stop_gradient=False)
    x = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)
    y = paddle.to_tensor(np_y, dtype=dtype, stop_gradient=False)
    z = paddle.to_tensor(np_z, dtype=dtype, stop_gradient=False)
    if use_phi:
        out = multi_out_module.discrete_out(w, x, y, z)
    else:
        out = w * 1 + x * 2 + y * 3 + z * 4

    out.backward()
    return out.numpy(), w.grad.numpy(), y.grad.numpy()


def discrete_out_static(use_phi, device, dtype, np_w, np_x, np_y, np_z):
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            w = static.data(name="w", shape=[None, np_x.shape[1]], dtype=dtype)
            x = static.data(name="x", shape=[None, np_x.shape[1]], dtype=dtype)
            y = static.data(name="y", shape=[None, np_y.shape[1]], dtype=dtype)
            z = static.data(name="z", shape=[None, np_z.shape[1]], dtype=dtype)
            w.stop_gradient = False
            x.stop_gradient = False
            y.stop_gradient = False
            z.stop_gradient = False
            if use_phi:
                out = multi_out_module.discrete_out(w, x, y, z)
            else:
                out = w * 1 + x * 2 + y * 3 + z * 4
            static.append_backward(out)

            exe = static.Executor()
            exe.run(static.default_startup_program())

            out_v, w_grad_v, y_grad_v = exe.run(
                static.default_main_program(),
                feed={
                    "w": np_w.astype(dtype),
                    "x": np_x.astype(dtype),
                    "y": np_y.astype(dtype),
                    "z": np_z.astype(dtype),
                },
                fetch_list=[
                    out.name,
                    w.name + "@GRAD",
                    y.name + "@GRAD",
                ],
            )
    paddle.disable_static()
    return out_v, w_grad_v, y_grad_v


class TestMultiOutputDtypes(unittest.TestCase):
    def setUp(self):
        self.custom_op = multi_out_module.multi_out
        self.dtypes = ['float32', 'float64']
        self.devices = ['cpu']
        self.np_w = np.random.uniform(-1, 1, [4, 8]).astype("float32")
        self.np_x = np.random.uniform(-1, 1, [4, 8]).astype("float32")
        self.np_y = np.random.uniform(-1, 1, [4, 8]).astype("float32")
        self.np_z = np.random.uniform(-1, 1, [4, 8]).astype("float32")

    def check_output(self, out, pd_out, name):
        np.testing.assert_array_equal(
            out,
            pd_out,
            err_msg='custom op {}: {},\n paddle api {}: {}'.format(
                name, out, name, pd_out
            ),
        )

    def run_static(self, device, dtype):
        paddle.set_device(device)
        x_data = np.random.uniform(-1, 1, [4, 8]).astype(dtype)

        with paddle.static.scope_guard(paddle.static.Scope()):
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data(name='X', shape=[None, 8], dtype=dtype)
                outs = self.custom_op(x)

                exe = paddle.static.Executor()
                exe.run(paddle.static.default_startup_program())
                res = exe.run(
                    paddle.static.default_main_program(),
                    feed={'X': x_data},
                    fetch_list=outs,
                )

                return res

    def check_multi_outputs(self, outs, is_dynamic=False):
        out, zero_float64, one_int32 = outs
        if is_dynamic:
            zero_float64 = zero_float64.numpy()
            one_int32 = one_int32.numpy()
        # Fake_float64
        self.assertTrue('float64' in str(zero_float64.dtype))
        np.testing.assert_array_equal(
            zero_float64, np.zeros([4, 8]).astype('float64')
        )
        # ZFake_int32
        self.assertTrue('int32' in str(one_int32.dtype))
        np.testing.assert_array_equal(
            one_int32, np.ones([4, 8]).astype('int32')
        )

    def test_multi_out_static(self):
        paddle.enable_static()
        for device in self.devices:
            for dtype in self.dtypes:
                res = self.run_static(device, dtype)
                self.check_multi_outputs(res)
        paddle.disable_static()

    def test_multi_out_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                paddle.set_device(device)
                x_data = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                x = paddle.to_tensor(x_data)
                outs = self.custom_op(x)

                self.assertTrue(len(outs) == 3)
                self.check_multi_outputs(outs, True)

    def test_discrete_out_static(self):
        for device in self.devices:
            for dtype in self.dtypes:
                (pd_out, pd_w_grad, pd_y_grad,) = discrete_out_static(
                    False,
                    device,
                    dtype,
                    self.np_w,
                    self.np_x,
                    self.np_y,
                    self.np_z,
                )
                (phi_out, phi_w_grad, phi_y_grad,) = discrete_out_static(
                    True,
                    device,
                    dtype,
                    self.np_w,
                    self.np_x,
                    self.np_y,
                    self.np_z,
                )
                self.check_output(phi_out, pd_out, "out")
                # NOTE: In static mode, the output gradient of custom operator has been optimized to shape=[1]. However, native paddle op's output shape = [4, 8], hence we need to fetch pd_w_grad[0][0] (By the way, something wrong with native paddle's gradient, the outputs with other indexes instead of pd_w_grad[0][0] is undefined in this unittest.)
                self.check_output(phi_w_grad, pd_w_grad[0][0], "w_grad")
                self.check_output(phi_y_grad, pd_y_grad[0][0], "y_grad")

    def test_discrete_out_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                (pd_out, pd_w_grad, pd_y_grad,) = discrete_out_dynamic(
                    False,
                    device,
                    dtype,
                    self.np_w,
                    self.np_x,
                    self.np_y,
                    self.np_z,
                )
                (phi_out, phi_w_grad, phi_y_grad,) = discrete_out_dynamic(
                    True,
                    device,
                    dtype,
                    self.np_w,
                    self.np_x,
                    self.np_y,
                    self.np_z,
                )
                self.check_output(phi_out, pd_out, "out")
                self.check_output(phi_w_grad, pd_w_grad, "w_grad")
                self.check_output(phi_y_grad, pd_y_grad, "y_grad")


if __name__ == '__main__':
    unittest.main()
