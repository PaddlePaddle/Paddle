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
import paddle.static as static
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
                fetch_list=[out.name],
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
        # install general extension
        # compile, install the custom op egg into site-packages under background
        cmd = 'cd {} && {} cpp_extension_setup.py install'.format(
            cur_dir, sys.executable
        )
        run_cmd(cmd)

        site_dir = site.getsitepackages()[0]
        custom_egg_path = [
            x for x in os.listdir(site_dir) if 'custom_cpp_extension' in x
        ]
        assert len(custom_egg_path) == 1, "Matched egg number is %d." % len(
            custom_egg_path
        )
        sys.path.append(os.path.join(site_dir, custom_egg_path[0]))

        # install mixed custom_op and extension
        cmd = 'cd {} && {} mix_relu_and_extension_setup.py install'.format(
            cur_dir, sys.executable
        )
        run_cmd(cmd)

        site_dir = site.getsitepackages()[0]
        custom_egg_path = [
            x for x in os.listdir(site_dir) if 'mix_relu_extension' in x
        ]
        assert len(custom_egg_path) == 1, "Matched egg number is %d." % len(
            custom_egg_path
        )
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
        self._test_extension_function_plain()
        self._test_extension_function_mixed()
        self._test_extension_class()
        self._test_nullable_tensor()
        self._test_optional_tensor()
        # Custom op
        self._test_static()
        self._test_dynamic()
        self._test_double_grad_dynamic()

    def _test_extension_function_plain(self):
        import custom_cpp_extension

        for dtype in self.dtypes:
            np_x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            x = paddle.to_tensor(np_x, dtype=dtype)
            np_y = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            y = paddle.to_tensor(np_y, dtype=dtype)
            # Test custom_cpp_extension
            out = custom_cpp_extension.custom_add(x, y)
            target_out = np.exp(np_x) + np.exp(np_y)
            np.testing.assert_allclose(out.numpy(), target_out, atol=1e-5)

            # Test we can call a method not defined in the main C++ file.
            out = custom_cpp_extension.custom_sub(x, y)
            target_out = np.exp(np_x) - np.exp(np_y)
            np.testing.assert_allclose(out.numpy(), target_out, atol=1e-5)

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

    def _test_extension_class(self):
        import custom_cpp_extension

        for dtype in self.dtypes:
            # Test custom_cpp_extension
            # Test we can use CppExtension class with C++ methods.
            power = custom_cpp_extension.Power(3, 3)
            self.assertEqual(power.get().sum(), 9)
            self.assertEqual(power.forward().sum(), 9)

            np_x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            x = paddle.to_tensor(np_x, dtype=dtype)

            power = custom_cpp_extension.Power(x)
            np.testing.assert_allclose(
                power.get().sum().numpy(), np.sum(np_x), atol=1e-5
            )
            np.testing.assert_allclose(
                power.forward().sum().numpy(),
                np.sum(np.power(np_x, 2)),
                atol=1e-5,
            )

    def _test_nullable_tensor(self):
        import custom_cpp_extension

        x = custom_cpp_extension.nullable_tensor(True)
        assert x is None, "Return None when input parameter return_none = True"
        x = custom_cpp_extension.nullable_tensor(False).numpy()
        x_np = np.ones(shape=[2, 2])
        np.testing.assert_array_equal(
            x,
            x_np,
            err_msg='extension out: {},\n numpy out: {}'.format(x, x_np),
        )

    def _test_optional_tensor(self):
        import custom_cpp_extension

        x = custom_cpp_extension.optional_tensor(True)
        assert (
            x is None
        ), "Return None when input parameter return_option = True"
        x = custom_cpp_extension.optional_tensor(False).numpy()
        x_np = np.ones(shape=[2, 2])
        np.testing.assert_array_equal(
            x,
            x_np,
            err_msg='extension out: {},\n numpy out: {}'.format(x, x_np),
        )

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
                err_msg='custom op out: {},\n paddle api out: {}'.format(
                    out, pd_out
                ),
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
                err_msg='custom op out: {},\n paddle api out: {}'.format(
                    out, pd_out
                ),
            )
            np.testing.assert_array_equal(
                x_grad,
                pd_x_grad,
                err_msg='custom op x grad: {},\n paddle api x grad: {}'.format(
                    x_grad, pd_x_grad
                ),
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
                err_msg='custom op out: {},\n paddle api out: {}'.format(
                    out, pd_out
                ),
            )
            np.testing.assert_array_equal(
                dx_grad,
                pd_dx_grad,
                err_msg='custom op dx grad: {},\n paddle api dx grad: {}'.format(
                    dx_grad, pd_dx_grad
                ),
            )


if __name__ == '__main__':
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        # only support Linux now
        exit()
    unittest.main()
