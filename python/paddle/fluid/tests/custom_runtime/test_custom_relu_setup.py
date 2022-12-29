# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import time
import unittest

import numpy as np

import paddle
import paddle.static as static
from paddle.fluid.framework import _test_eager_guard, in_dygraph_mode
from paddle.vision.transforms import Compose, Normalize


def custom_relu_dynamic(func, device, dtype, np_x, use_func=True):
    paddle.set_device(device)

    t = paddle.to_tensor(np_x, dtype=dtype)
    t.stop_gradient = False
    sys.stdout.flush()

    out = func(t) if use_func else paddle.nn.functional.relu(t)
    out.stop_gradient = False

    out.backward()

    if t.grad is None:
        return out.numpy(), t.grad
    else:
        return out.numpy(), t.grad.numpy()


def custom_relu_static(func, device, dtype, np_x, use_func=True):
    paddle.enable_static()
    print("DEBUG device ", device)
    paddle.set_device(device)

    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name="X", shape=[None, 8], dtype=dtype)
            x.stop_gradient = False
            out = func(x) if use_func else paddle.nn.functional.relu(x)
            static.append_backward(out)

            exe = static.Executor()
            exe.run(static.default_startup_program())
            # in static mode, x data has been covered by out
            out_v = exe.run(
                static.default_main_program(),
                feed={"X": np_x},
                fetch_list=[out.name],
            )

    paddle.disable_static()
    return out_v


def custom_relu_static_pe(func, device, dtype, np_x, use_func=True):
    paddle.enable_static()
    paddle.set_device(device)

    places = paddle.CustomPlace("custom_cpu", 0)

    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name="X", shape=[None, 8], dtype=dtype)
            x.stop_gradient = False
            out = func(x) if use_func else paddle.nn.functional.relu(x)
            static.append_backward(out)

            exe = static.Executor()
            exe.run(static.default_startup_program())

            # in static mode, x data has been covered by out
            compiled_prog = static.CompiledProgram(
                static.default_main_program()
            ).with_data_parallel(loss_name=out.name, places=places)
            out_v = exe.run(
                compiled_prog, feed={"X": np_x}, fetch_list=[out.name]
            )

    paddle.disable_static()
    return out_v


def custom_relu_double_grad_dynamic(func, device, dtype, np_x, use_func=True):
    paddle.set_device(device)

    t = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)

    out = func(t) if use_func else paddle.nn.functional.relu(t)
    out.stop_gradient = False
    dx = paddle.grad(
        outputs=[out], inputs=[t], create_graph=True, retain_graph=True
    )
    if in_dygraph_mode():
        dx[0].retain_grads()
    dx[0].backward()

    # Actually, dx[0].grad is an intermediate tensor of double_grad, we cannot obtain out_grad_grad
    assert dx[0].grad is not None
    return dx[0].numpy(), dx[0].grad.numpy()


class TestNewCustomOpSetUpInstall(unittest.TestCase):
    def setUp(self):
        self.time_start = time.time()
        # compile so and set to current path
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.temp_dir = tempfile.TemporaryDirectory()
        print("cur_dir", cur_dir)
        print("self.temp_dir", self.temp_dir.name)
        cmd = 'cd {} \
            && git clone {} \
            && cd PaddleCustomDevice \
            && git fetch origin \
            && git checkout {} -b dev \
            && cd backends/custom_cpu \
            && mkdir build && cd build && cmake .. && make -j8 \
            && pip install dist/paddle_custom_cpu*.whl \
            && python -c "import paddle; print(paddle.device.get_all_custom_device_type())" \
            && cd {} && {} custom_device_relu_setup.py install'.format(
            self.temp_dir.name,
            os.getenv('PLUGIN_URL'),
            os.getenv('PLUGIN_TAG'),
            cur_dir,
            sys.executable,
        )
        os.system(cmd)
        print("Custom device installed & custom_relu_setup installed")
        print(paddle.device.get_all_custom_device_type())

        # set environment for loading and registering compiled custom kernels
        # only valid in current process
        os.environ['CUSTOM_DEVICE_ROOT'] = os.path.join(
            cur_dir,
            '{}/PaddleCustomDevice/backends/custom_cpu/build'.format(
                self.temp_dir.name
            ),
        )
        print(paddle.device.get_all_custom_device_type())

        site_dir = site.getsitepackages()[0]
        print("site_dir", site_dir)
        custom_egg_path = [
            x
            for x in os.listdir(site_dir)
            if "custom_device_relu_module_setup" in x
        ]
        assert len(custom_egg_path) == 1, "Matched egg number is %d." % len(
            custom_egg_path
        )
        print("custom_egg_path", custom_egg_path[0])
        sys.path.append(os.path.join(site_dir, custom_egg_path[0]))

        # usage: import the package directly
        import custom_device_relu_module_setup

        self.custom_op = custom_device_relu_module_setup.custom_relu

        self.dtypes = ["float32", "float64"]
        self.devices = ["custom_cpu"]

        # config seed
        SEED = 2021
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)
        print("DEBUG setup finished")
        print("SETUP TIME", time.time() - self.time_start)

    def tearDown(self):
        print("TearDown TIME", time.time() - self.time_start)
        print("DEBUG begin tearDown")
        os.system("pip uninstall paddle_custom_cpu")
        print("DEBUG uninstall paddle_custom_cpu finish")
        self.temp_dir.cleanup()
        del os.environ['CUSTOM_DEVICE_ROOT']
        print("DEBUG finish tearDown")
        print("TearDown TIME end", time.time() - self.time_start)

    def test_custom_device(self):
        print("DEBUG begin test ")
        self._test_static()
        print("DEBUG finish test _test_static")
        self._test_static_pe()
        print("DEBUG finish test _test_static_pe")
        self._test_dynamic()
        print("DEBUG finish test _test_dynamic")
        self._test_double_grad_dynamic()
        print("DEBUG finish test _test_double_grad_dynamic")
        self._test_with_dataloader()
        print("DEBUG finish test _test_with_dataloader")

    def _test_static(self):
        for device in self.devices:
            for dtype in self.dtypes:
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                out = custom_relu_static(self.custom_op, device, dtype, x)
                pd_out = custom_relu_static(
                    self.custom_op, device, dtype, x, False
                )
                np.testing.assert_array_equal(
                    out,
                    pd_out,
                    err_msg="custom op out: {},\n paddle api out: {}".format(
                        out, pd_out
                    ),
                )

    def _test_static_pe(self):
        for device in self.devices:
            for dtype in self.dtypes:
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                out = custom_relu_static_pe(self.custom_op, device, dtype, x)
                pd_out = custom_relu_static_pe(
                    self.custom_op, device, dtype, x, False
                )
                np.testing.assert_array_equal(
                    out,
                    pd_out,
                    err_msg="custom op out: {},\n paddle api out: {}".format(
                        out, pd_out
                    ),
                )

    def func_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                out, x_grad = custom_relu_dynamic(
                    self.custom_op, device, dtype, x
                )
                pd_out, pd_x_grad = custom_relu_dynamic(
                    self.custom_op, device, dtype, x, False
                )
                np.testing.assert_array_equal(
                    out,
                    pd_out,
                    err_msg="custom op out: {},\n paddle api out: {}".format(
                        out, pd_out
                    ),
                )
                np.testing.assert_array_equal(
                    x_grad,
                    pd_x_grad,
                    err_msg="custom op x grad: {},\n paddle api x grad: {}".format(
                        x_grad, pd_x_grad
                    ),
                )

    def _test_dynamic(self):
        with _test_eager_guard():
            self.func_dynamic()
        self.func_dynamic()

    def func_double_grad_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                out, dx_grad = custom_relu_double_grad_dynamic(
                    self.custom_op, device, dtype, x
                )
                pd_out, pd_dx_grad = custom_relu_double_grad_dynamic(
                    self.custom_op, device, dtype, x, False
                )
                np.testing.assert_array_equal(
                    out,
                    pd_out,
                    err_msg="custom op out: {},\n paddle api out: {}".format(
                        out, pd_out
                    ),
                )
                np.testing.assert_array_equal(
                    dx_grad,
                    pd_dx_grad,
                    err_msg="custom op dx grad: {},\n paddle api dx grad: {}".format(
                        dx_grad, pd_dx_grad
                    ),
                )

    def _test_double_grad_dynamic(self):
        with _test_eager_guard():
            self.func_double_grad_dynamic()
        self.func_double_grad_dynamic()

    def _test_with_dataloader(self):
        for device in self.devices:
            paddle.set_device(device)
            # data loader
            transform = Compose(
                [Normalize(mean=[127.5], std=[127.5], data_format="CHW")]
            )
            train_dataset = paddle.vision.datasets.MNIST(
                mode="train", transform=transform
            )
            train_loader = paddle.io.DataLoader(
                train_dataset,
                batch_size=64,
                shuffle=True,
                drop_last=True,
                num_workers=0,
            )

            for batch_id, (image, _) in enumerate(train_loader()):
                out = self.custom_op(image)
                pd_out = paddle.nn.functional.relu(image)
                np.testing.assert_array_equal(
                    out,
                    pd_out,
                    err_msg="custom op out: {},\n paddle api out: {}".format(
                        out, pd_out
                    ),
                )

                if batch_id == 5:
                    break


if __name__ == "__main__":
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        # only support Linux now
        exit()
    unittest.main()
