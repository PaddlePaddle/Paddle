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
import sys
import tempfile
import unittest
from site import getsitepackages

import numpy as np


def custom_relu_dynamic(func, device, dtype, np_x, use_func=True):
    import paddle

    paddle.set_device(device)

    t = paddle.to_tensor(np_x, dtype=dtype)
    t.stop_gradient = False
    t.retain_grads()
    sys.stdout.flush()

    out = func(t) if use_func else paddle.nn.functional.relu(t)
    out.stop_gradient = False

    out.backward()

    if t.grad is None:
        return out.numpy(), t.grad
    else:
        return out.numpy(), t.grad.numpy()


def custom_relu_static(func, device, dtype, np_x, use_func=True):
    import paddle
    from paddle import static

    paddle.enable_static()
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
                fetch_list=[out],
            )

    paddle.disable_static()
    return out_v


def custom_relu_double_grad_dynamic(func, device, dtype, np_x, use_func=True):
    import paddle

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


class TestNewCustomOpSetUpInstall(unittest.TestCase):
    def setUp(self):
        # compile so and set to current path
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.temp_dir = tempfile.TemporaryDirectory()
        cmd = 'cd {} \
            && git clone --depth 1 {} \
            && cd PaddleCustomDevice \
            && git fetch origin \
            && git checkout {} -b dev \
            && cd backends/custom_cpu \
            && mkdir build && cd build && cmake .. -DPython_EXECUTABLE={} -DWITH_TESTING=OFF && make -j8 \
            && cd {}'.format(
            self.temp_dir.name,
            os.getenv('PLUGIN_URL'),
            os.getenv('PLUGIN_TAG'),
            sys.executable,
            self.cur_dir,
        )
        os.system(cmd)

        # set environment for loading and registering compiled custom kernels
        # only valid in current process
        os.environ['CUSTOM_DEVICE_ROOT'] = os.path.join(
            self.cur_dir,
            f'{self.temp_dir.name}/PaddleCustomDevice/backends/custom_cpu/build',
        )

        # `import paddle` loads custom_cpu.so, hence we must import paddle after finishing build PaddleCustomDevice
        import paddle

        # [Why specific paddle_includes directory?]
        # Add paddle_includes to pass CI, for more details,
        # please refer to the comments in `paddle/tests/custom_op/utils.py``
        paddle_includes = []
        for site_packages_path in getsitepackages():
            paddle_includes.append(
                os.path.join(site_packages_path, 'paddle', 'include')
            )
            paddle_includes.append(
                os.path.join(
                    site_packages_path, 'paddle', 'include', 'third_party'
                )
            )

        custom_module = paddle.utils.cpp_extension.load(
            name='custom_device',
            sources=['custom_op.cc'],
            extra_include_paths=paddle_includes,  # add for Coverage CI
            extra_cxx_cflags=["-w", "-g"],  # test for cc flags
            # build_directory=self.cur_dir,
            verbose=True,
        )
        self.custom_op = custom_module.custom_relu
        self.custom_stream_op = custom_module.custom_stream

        self.dtypes = ["float32", "float64"]
        self.device = "custom_cpu"

        # config seed
        SEED = 2021
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

    def tearDown(self):
        self.temp_dir.cleanup()
        del os.environ['CUSTOM_DEVICE_ROOT']

    def test_custom_device(self):
        self._test_static()
        self._test_dynamic()
        self._test_double_grad_dynamic()
        self._test_with_dataloader()
        self._test_stream()

    def _test_static(self):
        for dtype in self.dtypes:
            x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            out = custom_relu_static(self.custom_op, self.device, dtype, x)
            pd_out = custom_relu_static(
                self.custom_op, self.device, dtype, x, False
            )
            np.testing.assert_array_equal(
                out,
                pd_out,
                err_msg=f"custom op out: {out},\n paddle api out: {pd_out}",
            )

    def _test_dynamic(self):
        for dtype in self.dtypes:
            x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            out, x_grad = custom_relu_dynamic(
                self.custom_op, self.device, dtype, x
            )
            pd_out, pd_x_grad = custom_relu_dynamic(
                self.custom_op, self.device, dtype, x, False
            )
            np.testing.assert_array_equal(
                out,
                pd_out,
                err_msg=f"custom op out: {out},\n paddle api out: {pd_out}",
            )
            np.testing.assert_array_equal(
                x_grad,
                pd_x_grad,
                err_msg=f"custom op x grad: {x_grad},\n paddle api x grad: {pd_x_grad}",
            )

    def _test_double_grad_dynamic(self):
        for dtype in self.dtypes:
            x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
            out, dx_grad = custom_relu_double_grad_dynamic(
                self.custom_op, self.device, dtype, x
            )
            pd_out, pd_dx_grad = custom_relu_double_grad_dynamic(
                self.custom_op, self.device, dtype, x, False
            )
            np.testing.assert_array_equal(
                out,
                pd_out,
                err_msg=f"custom op out: {out},\n paddle api out: {pd_out}",
            )
            np.testing.assert_array_equal(
                dx_grad,
                pd_dx_grad,
                err_msg=f"custom op dx grad: {dx_grad},\n paddle api dx grad: {pd_dx_grad}",
            )

    def _test_with_dataloader(self):
        import paddle
        from paddle.vision.transforms import Compose, Normalize

        paddle.set_device(self.device)
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
                err_msg=f"custom op out: {out},\n paddle api out: {pd_out}",
            )

            if batch_id == 5:
                break

    def _test_stream(self):
        import paddle

        paddle.set_device(self.device)
        x = paddle.ones([2, 2], dtype='float32')
        out = self.custom_stream_op(x)

        np.testing.assert_array_equal(x.numpy(), out.numpy())


if __name__ == "__main__":
    if os.name == 'nt' or sys.platform.startswith('darwin'):
        # only support Linux now
        sys.exit()
    unittest.main()
