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
from paddle.fluid.framework import _test_eager_guard


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


def custom_relu_static(func,
                       device,
                       dtype,
                       np_x,
                       use_func=True,
                       test_infer=False):
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
            # in static mode, x data has been covered by out
            out_v = exe.run(static.default_main_program(),
                            feed={'X': np_x},
                            fetch_list=[out.name])

    paddle.disable_static()
    return out_v


def custom_relu_static_pe(func, device, dtype, np_x, use_func=True):
    paddle.enable_static()
    paddle.set_device(device)

    places = static.cpu_places() if device is 'cpu' else static.cuda_places()
    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            x = static.data(name='X', shape=[None, 8], dtype=dtype)
            x.stop_gradient = False
            out = func(x) if use_func else paddle.nn.functional.relu(x)
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


def custom_relu_static_inference(func, device, np_data, np_label, path_prefix):
    paddle.set_device(device)

    with static.scope_guard(static.Scope()):
        with static.program_guard(static.Program()):
            # simple module
            data = static.data(
                name='data', shape=[None, 1, 28, 28], dtype='float32')
            label = static.data(name='label', shape=[None, 1], dtype='int64')

            hidden = static.nn.fc(data, size=128)
            hidden = func(hidden)
            hidden = static.nn.fc(hidden, size=128)
            predict = static.nn.fc(hidden, size=10, activation='softmax')
            loss = paddle.nn.functional.cross_entropy(input=hidden, label=label)
            avg_loss = paddle.mean(loss)

            opt = paddle.optimizer.SGD(learning_rate=0.1)
            opt.minimize(avg_loss)

            # run start up model
            exe = static.Executor()
            exe.run(static.default_startup_program())

            # train
            for i in range(4):
                avg_loss_v = exe.run(static.default_main_program(),
                                     feed={'data': np_data,
                                           'label': np_label},
                                     fetch_list=[avg_loss])

            # save inference model
            static.save_inference_model(path_prefix, [data], [predict], exe)

            # get train predict value
            predict_v = exe.run(static.default_main_program(),
                                feed={'data': np_data,
                                      'label': np_label},
                                fetch_list=[predict])

    return predict_v


def custom_relu_double_grad_dynamic(func, device, dtype, np_x, use_func=True):
    paddle.set_device(device)

    t = paddle.to_tensor(np_x, dtype=dtype, stop_gradient=False)

    out = func(t) if use_func else paddle.nn.functional.relu(t)
    out.stop_gradient = False

    dx = paddle.grad(
        outputs=[out], inputs=[t], create_graph=True, retain_graph=True)

    dx[0].backward()

    assert dx[0].grad is not None
    return dx[0].numpy(), dx[0].grad.numpy()


class TestNewCustomOpSetUpInstall(unittest.TestCase):
    def setUp(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        # compile, install the custom op egg into site-packages under background
        if os.name == 'nt':
            cmd = 'cd /d {} && python custom_relu_setup.py install'.format(
                cur_dir)
        else:
            cmd = 'cd {} && {} custom_relu_setup.py install'.format(
                cur_dir, sys.executable)
        run_cmd(cmd)

        # NOTE(Aurelius84): Normally, it's no need to add following codes for users.
        # But we simulate to pip install in current process, so interpreter don't snap
        # sys.path has been updated. So we update it manually.

        # See: https://stackoverflow.com/questions/56974185/import-runtime-installed-module-using-pip-in-python-3
        if os.name == 'nt':
            # NOTE(zhouwei25): getsitepackages on windows will return a list: [python install dir, site packages dir]
            site_dir = site.getsitepackages()[1]
        else:
            site_dir = site.getsitepackages()[0]
        custom_egg_path = [
            x for x in os.listdir(site_dir) if 'custom_relu_module_setup' in x
        ]
        assert len(custom_egg_path) == 1, "Matched egg number is %d." % len(
            custom_egg_path)
        sys.path.append(os.path.join(site_dir, custom_egg_path[0]))

        # usage: import the package directly
        import custom_relu_module_setup
        # `custom_relu_dup` is same as `custom_relu_dup`
        self.custom_ops = [
            custom_relu_module_setup.custom_relu,
            custom_relu_module_setup.custom_relu_dup
        ]

        self.dtypes = ['float32', 'float64']
        if paddle.is_compiled_with_cuda():
            self.dtypes.append('float16')
        self.devices = ['cpu']
        if paddle.is_compiled_with_cuda():
            self.devices.append('gpu')

        # config seed
        SEED = 2021
        paddle.seed(SEED)
        paddle.framework.random._manual_program_seed(SEED)

    def test_static(self):
        for device in self.devices:
            for dtype in self.dtypes:
                if device == 'cpu' and dtype == 'float16':
                    continue
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                for custom_op in self.custom_ops:
                    out = custom_relu_static(custom_op, device, dtype, x)
                    pd_out = custom_relu_static(custom_op, device, dtype, x,
                                                False)
                    self.assertTrue(
                        np.array_equal(out, pd_out),
                        "custom op out: {},\n paddle api out: {}".format(
                            out, pd_out))

    def test_static_pe(self):
        for device in self.devices:
            for dtype in self.dtypes:
                if device == 'cpu' and dtype == 'float16':
                    continue
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                for custom_op in self.custom_ops:
                    out = custom_relu_static_pe(custom_op, device, dtype, x)
                    pd_out = custom_relu_static_pe(custom_op, device, dtype, x,
                                                   False)
                    self.assertTrue(
                        np.array_equal(out, pd_out),
                        "custom op out: {},\n paddle api out: {}".format(
                            out, pd_out))

    def func_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                if device == 'cpu' and dtype == 'float16':
                    continue
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                for custom_op in self.custom_ops:
                    out, x_grad = custom_relu_dynamic(custom_op, device, dtype,
                                                      x)
                    pd_out, pd_x_grad = custom_relu_dynamic(custom_op, device,
                                                            dtype, x, False)
                    self.assertTrue(
                        np.array_equal(out, pd_out),
                        "custom op out: {},\n paddle api out: {}".format(
                            out, pd_out))
                    self.assertTrue(
                        np.array_equal(x_grad, pd_x_grad),
                        "custom op x grad: {},\n paddle api x grad: {}".format(
                            x_grad, pd_x_grad))

    def test_dynamic(self):
        with _test_eager_guard():
            self.func_dynamic()
        self.func_dynamic()

    def test_static_save_and_load_inference_model(self):
        paddle.enable_static()
        np_data = np.random.random((1, 1, 28, 28)).astype("float32")
        np_label = np.random.random((1, 1)).astype("int64")
        path_prefix = "custom_op_inference/custom_relu"
        for device in self.devices:
            predict = custom_relu_static_inference(
                self.custom_ops[0], device, np_data, np_label, path_prefix)
            # load inference model
            with static.scope_guard(static.Scope()):
                exe = static.Executor()
                [inference_program, feed_target_names,
                 fetch_targets] = static.load_inference_model(path_prefix, exe)
                predict_infer = exe.run(inference_program,
                                        feed={feed_target_names[0]: np_data},
                                        fetch_list=fetch_targets)
                self.assertTrue(
                    np.array_equal(predict, predict_infer),
                    "custom op predict: {},\n custom op infer predict: {}".
                    format(predict, predict_infer))
        paddle.disable_static()

    def test_static_save_and_run_inference_predictor(self):
        paddle.enable_static()
        np_data = np.random.random((1, 1, 28, 28)).astype("float32")
        np_label = np.random.random((1, 1)).astype("int64")
        path_prefix = "custom_op_inference/custom_relu"
        from paddle.inference import Config
        from paddle.inference import create_predictor
        for device in self.devices:
            predict = custom_relu_static_inference(
                self.custom_ops[0], device, np_data, np_label, path_prefix)
            # load inference model
            config = Config(path_prefix + ".pdmodel",
                            path_prefix + ".pdiparams")
            predictor = create_predictor(config)
            input_tensor = predictor.get_input_handle(predictor.get_input_names(
            )[0])
            input_tensor.reshape(np_data.shape)
            input_tensor.copy_from_cpu(np_data.copy())
            predictor.run()
            output_tensor = predictor.get_output_handle(
                predictor.get_output_names()[0])
            predict_infer = output_tensor.copy_to_cpu()
            self.assertTrue(
                np.isclose(
                    predict, predict_infer, rtol=5e-5).any(),
                "custom op predict: {},\n custom op infer predict: {}".format(
                    predict, predict_infer))
        paddle.disable_static()

    def test_func_double_grad_dynamic(self):
        for device in self.devices:
            for dtype in self.dtypes:
                if device == 'cpu' and dtype == 'float16':
                    continue
                x = np.random.uniform(-1, 1, [4, 8]).astype(dtype)
                out, dx_grad = custom_relu_double_grad_dynamic(
                    self.custom_ops[0], device, dtype, x)
                pd_out, pd_dx_grad = custom_relu_double_grad_dynamic(
                    self.custom_ops[0], device, dtype, x, False)
                self.assertTrue(
                    np.array_equal(out, pd_out),
                    "custom op out: {},\n paddle api out: {}".format(out,
                                                                     pd_out))
                self.assertTrue(
                    np.array_equal(dx_grad, pd_dx_grad),
                    "custom op dx grad: {},\n paddle api dx grad: {}".format(
                        dx_grad, pd_dx_grad))


if __name__ == '__main__':
    unittest.main()
