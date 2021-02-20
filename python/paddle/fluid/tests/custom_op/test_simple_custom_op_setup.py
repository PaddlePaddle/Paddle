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
import sys
import site
import unittest
import paddle
import paddle.static as static
import subprocess
import numpy as np
from paddle.utils.cpp_extension.extension_utils import run_cmd


def relu2_dynamic(func, device, dtype, np_x, use_func=True):
    paddle.set_device(device)

    t = paddle.to_tensor(np_x)
    t.stop_gradient = False

    out = func(t)[0] if use_func else paddle.nn.functional.relu(t)
    out.stop_gradient = False

    out.backward()

    return out.numpy(), t.grad


def relu2_static(func, device, dtype, np_x, use_func=True):
    paddle.enable_static()
    paddle.set_device(device)

    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name='X', shape=[None, 8], dtype=dtype)
            x.stop_gradient = False
            # out, fake_float64, fake_int32
            out = func(x)[0] if use_func else paddle.nn.functional.relu(x)
            static.append_backward(out)

            exe = static.Executor()
            exe.run(static.default_startup_program())
            # in static mode, x data has been covered by out
            out_v = exe.run(static.default_main_program(),
                            feed={'X': np_x},
                            fetch_list=[out.name])

    paddle.disable_static()
    return out_v


def relu2_static_pe(func, device, dtype, np_x, use_func=True):
    paddle.enable_static()
    paddle.set_device(device)

    places = static.cpu_places() if device is 'cpu' else static.cuda_places()
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name='X', shape=[None, 8], dtype=dtype)
            x.stop_gradient = False
            out = func(x)[0] if use_func else paddle.nn.functional.relu(x)
            static.append_backward(out)

            exe = static.Executor()
            exe.run(static.default_startup_program())

            # in static mode, x data has been covered by out
            compiled_prog = static.CompiledProgram(static.default_main_program(
            )).with_data_parallel(
                loss_name=out.name, places=places)
            out_v = exe.run(compiled_prog,
                            feed={'X': np_x},
                            fetch_list=[out.name])

    paddle.disable_static()
    return out_v


class TestNewCustomOpSetUpInstall(unittest.TestCase):
    def setUp(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        # compile, install the custom op egg into site-packages under background
        cmd = 'cd {} && python setup_install_simple.py install'.format(cur_dir)
        run_cmd(cmd)

        # NOTE(Aurelius84): Normally, it's no need to add following codes for users.
        # But we simulate to pip install in current process, so interpreter don't snap
        # sys.path has been updated. So we update it manually.

        # See: https://stackoverflow.com/questions/56974185/import-runtime-installed-module-using-pip-in-python-3
        site_dir = site.getsitepackages()[0]
        custom_egg_path = [
            x for x in os.listdir(site_dir) if 'simple_setup_relu2' in x
        ]
        assert len(custom_egg_path) == 1, "Matched egg number is %d." % len(
            custom_egg_path)
        sys.path.append(os.path.join(site_dir, custom_egg_path[0]))

        # usage: import the package directly
        import simple_setup_relu2
        self.custom_ops = [simple_setup_relu2.relu2, simple_setup_relu2.relu3]

        self.dtypes = ['float32', 'float64']
        self.devices = ['cpu', 'gpu']

    def test_static(self):
        for device in self.devices:
            for dtype in self.dtypes:
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                for custom_op in self.custom_ops:
                    out = relu2_static(custom_op, device, dtype, x)
                    pd_out = relu2_static(custom_op, device, dtype, x, False)
                    self.assertTrue(
                        np.array_equal(out, pd_out),
                        "custom op out: {},\n paddle api out: {}".format(
                            out, pd_out))

    def test_static_pe(self):
        for device in self.devices:
            for dtype in self.dtypes:
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                for custom_op in self.custom_ops:
                    out = relu2_static_pe(custom_op, device, dtype, x)
                    pd_out = relu2_static_pe(custom_op, device, dtype, x, False)
                    self.assertTrue(
                        np.array_equal(out, pd_out),
                        "custom op out: {},\n paddle api out: {}".format(
                            out, pd_out))

    def test_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                for custom_op in self.custom_ops:
                    out, x_grad = relu2_dynamic(custom_op, device, dtype, x)
                    pd_out, pd_x_grad = relu2_dynamic(custom_op, device, dtype,
                                                      x, False)
                    self.assertTrue(
                        np.array_equal(out, pd_out),
                        "custom op out: {},\n paddle api out: {}".format(
                            out, pd_out))
                    self.assertTrue(
                        np.array_equal(x_grad, pd_x_grad),
                        "custom op x grad: {},\n paddle api x grad: {}".format(
                            x_grad, pd_x_grad))


if __name__ == '__main__':
    unittest.main()
