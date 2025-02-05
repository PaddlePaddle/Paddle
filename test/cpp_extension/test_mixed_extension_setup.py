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
import site
import sys
import unittest

import numpy as np

import paddle
from paddle import static
from paddle.utils.cpp_extension.extension_utils import run_cmd


def custom_relu_static(
    func, device, dtype, np_x, use_func=True, test_infer=False
):
    paddle.enable_static()
    paddle.set_device(device)

    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name='X', shape=[None, 8], dtype=dtype)
            x.stop_gradient = False
            out = func(x) if use_func else paddle.nn.functional.relu(x)
            static.append_backward(out)

            exe = static.Executor()
            exe.run(static.default_startup_program())
            # in static graph mode, x data has been covered by out
            out_v = exe.run(
                static.default_main_program(),
                feed={'X': np_x},
                fetch_list=[out],
            )

    paddle.disable_static()
    return out_v


def custom_relu_dynamic(func, device, dtype, np_x, use_func=True):
    paddle.set_device(device)

    t = paddle.to_tensor(np_x, dtype=dtype)
    t.stop_gradient = False

    out = func(t) if use_func else paddle.nn.functional.relu(t)
    out.stop_gradient = False

    out.backward()

    if t.grad is None:
        return out.numpy(), t.grad
    else:
        return out.numpy(), t.grad.numpy()


def custom_relu_double_grad_dynamic(func, device, dtype, np_x, use_func=True):
    paddle.set_device(device)

    t = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)
    t.retain_grads()

    out = func(t) if use_func else paddle.nn.functional.relu(t)
    out.retain_grads()
    dx = paddle.grad(
        outputs=out,
        inputs=t,
        grad_outputs=paddle.ones_like(t),
        create_graph=True,
        retain_graph=True,
    )

    ddout = paddle.grad(
        outputs=dx[0],
        inputs=out.grad,
        grad_outputs=paddle.ones_like(t),
        create_graph=False,
    )

    assert ddout[0].numpy() is not None
    return dx[0].numpy(), ddout[0].numpy()


class TestCppExtensionSetupInstall(unittest.TestCase):
    """
    Tests setup install cpp extensions.
    """

    def setUp(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        # install mixed custom_op and extension
        # compile, install the custom op egg into site-packages under background
        site_dir = site.getsitepackages()[0]
        cmd = f'cd {cur_dir} && {sys.executable} mix_relu_and_extension_setup.py install'
        if os.name != 'nt':
            cmd += f' --install-lib={site_dir}'
        run_cmd(cmd)

        custom_egg_path = [
            x for x in os.listdir(site_dir) if 'mix_relu_extension' in x
        ]
        assert (
            len(custom_egg_path) == 1
        ), f"Matched egg number is {len(custom_egg_path)}."
        sys.path.append(os.path.join(site_dir, custom_egg_path[0]))
        #################################

        # config seed
        SEED = 2021
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

        self.dtypes = ['float32', 'float64']

    def tearDown(self):
        pass

    def test_cpp_extension(self):
        # Extension
        self._test_extension_function_mixed()
        # Custom op
        self._test_static()
        self._test_dynamic()
        self._test_double_grad_dynamic()

    def _test_extension_function_mixed(self):
        import mix_relu_extension

        for dtype in self.dtypes:
            np_x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            x = paddle.to_tensor(np_x, dtype=dtype)
            np_y = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            y = paddle.to_tensor(np_y, dtype=dtype)

            # Test mix_relu_extension
            out = mix_relu_extension.custom_add2(x, y)
            target_out = np.exp(np_x) + np.exp(np_y)
            np.testing.assert_allclose(out.numpy(), target_out, atol=1e-5)

            # Test we can call a method not defined in the main C++ file.
            out = mix_relu_extension.custom_sub2(x, y)
            target_out = np.exp(np_x) - np.exp(np_y)
            np.testing.assert_allclose(out.numpy(), target_out, atol=1e-5)

    def _test_static(self):
        import mix_relu_extension

        for dtype in self.dtypes:
            x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            out = custom_relu_static(
                mix_relu_extension.custom_relu, "CPU", dtype, x
            )
            pd_out = custom_relu_static(
                mix_relu_extension.custom_relu, "CPU", dtype, x, False
            )
            np.testing.assert_array_equal(
                out,
                pd_out,
                err_msg=f'custom op out: {out},\n paddle api out: {pd_out}',
            )

    def _test_dynamic(self):
        import mix_relu_extension

        for dtype in self.dtypes:
            x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            out, x_grad = custom_relu_dynamic(
                mix_relu_extension.custom_relu, "CPU", dtype, x
            )
            pd_out, pd_x_grad = custom_relu_dynamic(
                mix_relu_extension.custom_relu, "CPU", dtype, x, False
            )
            np.testing.assert_array_equal(
                out,
                pd_out,
                err_msg=f'custom op out: {out},\n paddle api out: {pd_out}',
            )
            np.testing.assert_array_equal(
                x_grad,
                pd_x_grad,
                err_msg=f'custom op x grad: {x_grad},\n paddle api x grad: {pd_x_grad}',
            )

    def _test_double_grad_dynamic(self):
        import mix_relu_extension

        for dtype in self.dtypes:
            x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            out, dx_grad = custom_relu_double_grad_dynamic(
                mix_relu_extension.custom_relu, "CPU", dtype, x
            )
            pd_out, pd_dx_grad = custom_relu_double_grad_dynamic(
                mix_relu_extension.custom_relu, "CPU", dtype, x, False
            )
            np.testing.assert_array_equal(
                out,
                pd_out,
                err_msg=f'custom op out: {out},\n paddle api out: {pd_out}',
            )
            np.testing.assert_array_equal(
                dx_grad,
                pd_dx_grad,
                err_msg=f'custom op dx grad: {dx_grad},\n paddle api dx grad: {pd_dx_grad}',
            )


if __name__ == '__main__':
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        # only support Linux now
        sys.exit()
    unittest.main()
