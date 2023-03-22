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
from utils import extra_cc_args, extra_nvcc_args, paddle_includes

import paddle
import paddle.static as static
from paddle.utils.cpp_extension import get_build_directory, load
from paddle.utils.cpp_extension.extension_utils import run_cmd

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = '{}\\custom_optional\\custom_optional.pyd'.format(get_build_directory())
if os.name == 'nt' and os.path.isfile(file):
    cmd = 'del {}'.format(file)
    run_cmd(cmd, True)

# Compile and load custom op Just-In-Time.
custom_optional = load(
    name='custom_optional',
    sources=['custom_optional.cc'],
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cflags
    extra_cuda_cflags=extra_nvcc_args,  # test for cflags
    verbose=True,
)


def optional_dynamic_add(phi_func, device, dtype, np_x, np_y):
    paddle.set_device(device)
    x = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)

    if np_y is not None:
        y = paddle.to_tensor(np_y, dtype=dtype, stop_gradient=False)
    else:
        y = x
    if phi_func:
        out = custom_optional.custom_add(x, y if np_y is not None else None)
    else:
        out = paddle.add(x, y)

    out.backward()
    return x.numpy(), out.numpy(), x.grad.numpy()


def optional_static_add(phi_func, device, dtype, np_x, np_y):
    paddle.enable_static()
    paddle.set_device(device)
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name="x", shape=[None, np_x.shape[1]], dtype=dtype)
            x.stop_gradient = False
            if np_y is not None:
                y = static.data(
                    name="y", shape=[None, np_x.shape[1]], dtype=dtype
                )
                y.stop_gradient = False
                feed_dict = {
                    "x": np_x.astype(dtype),
                    "y": np_y.astype(dtype),
                }
            else:
                y = x
                feed_dict = {
                    "x": np_x.astype(dtype),
                }
            if phi_func:
                out = custom_optional.custom_add(
                    x, y if np_y is not None else None
                )
            else:
                out = paddle.add(x, y)

            mean_out = paddle.mean(out)
            static.append_backward(mean_out)

            exe = static.Executor()
            exe.run(static.default_startup_program())

            x_v, out_v, x_grad_v = exe.run(
                static.default_main_program(),
                feed=feed_dict,
                fetch_list=[
                    x.name,
                    out.name,
                    x.name + "@GRAD",
                ],
            )
    paddle.disable_static()
    return x_v, out_v, x_grad_v


class TestCustomOptionalJit(unittest.TestCase):
    def setUp(self):
        self.dtypes = ['float32', 'float64']
        self.devices = ['cpu']
        self.np_x = np.random.random((3, 2)).astype("float32")
        self.np_y = np.random.random((3, 2)).astype("float32")

    def check_output(self, out, pd_out, name):
        np.testing.assert_array_equal(
            out,
            pd_out,
            err_msg='custom op {}: {},\n paddle api {}: {}'.format(
                name, out, name, pd_out
            ),
        )

    def check_output_allclose(self, out, pd_out, name):
        np.testing.assert_allclose(
            out,
            pd_out,
            rtol=5e-5,
            atol=1e-2,
            err_msg='custom op {}: {},\n paddle api {}: {}'.format(
                name, out, name, pd_out
            ),
        )

    def test_static_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                (pd_x, pd_out, pd_x_grad,) = optional_static_add(
                    False,
                    device,
                    dtype,
                    self.np_x,
                    self.np_y,
                )
                (phi_x, phi_out, phi_x_grad,) = optional_static_add(
                    True,
                    device,
                    dtype,
                    self.np_x,
                    self.np_y,
                )
                self.check_output(phi_x, pd_x, "x")
                self.check_output(phi_out, pd_out, "out")
                self.check_output(phi_x_grad, pd_x_grad, "x_grad")

    def test_dynamic_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                (pd_x, pd_out, pd_x_grad,) = optional_dynamic_add(
                    False,
                    device,
                    dtype,
                    self.np_x,
                    self.np_y,
                )
                (phi_x, phi_out, phi_x_grad,) = optional_dynamic_add(
                    True,
                    device,
                    dtype,
                    self.np_x,
                    self.np_y,
                )

                self.check_output(phi_x, pd_x, "x")
                self.check_output(phi_out, pd_out, "out")
                self.check_output(phi_x_grad, pd_x_grad, "x_grad")

    def test_optional_static_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                (pd_x, pd_out, pd_x_grad,) = optional_static_add(
                    False,
                    device,
                    dtype,
                    self.np_x,
                    None,
                )
                (phi_x, phi_out, phi_x_grad,) = optional_static_add(
                    True,
                    device,
                    dtype,
                    self.np_x,
                    None,
                )

                self.check_output(phi_x, pd_x, "x")
                self.check_output(phi_out, pd_out, "out")
                self.check_output(phi_x_grad, pd_x_grad, "x_grad")

    def test_optional_dynamic_add(self):
        for device in self.devices:
            for dtype in self.dtypes:
                (pd_x, pd_out, pd_x_grad,) = optional_dynamic_add(
                    False,
                    device,
                    dtype,
                    self.np_x,
                    None,
                )
                (phi_x, phi_out, phi_x_grad,) = optional_dynamic_add(
                    True,
                    device,
                    dtype,
                    self.np_x,
                    None,
                )

                self.check_output(phi_x, pd_x, "x")
                self.check_output(phi_out, pd_out, "out")
                self.check_output(phi_x_grad, pd_x_grad, "x_grad")


if __name__ == "__main__":
    unittest.main()
