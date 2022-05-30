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
from paddle import nn
from paddle.utils.cpp_extension import load, get_build_directory
from paddle.utils.cpp_extension.extension_utils import run_cmd

from utils import paddle_includes, extra_cc_args, extra_nvcc_args, IS_MAC
from paddle.fluid.framework import _test_eager_guard, _in_legacy_dygraph

# Because Windows don't use docker, the shared lib already exists in the
# cache dir, it will not be compiled again unless the shared lib is removed.
file = '{}\\custom_relu_for_model_jit\\custom_relu_for_model_jit.pyd'.format(
    get_build_directory())
if os.name == 'nt' and os.path.isfile(file):
    cmd = 'del {}'.format(file)
    run_cmd(cmd, True)

# Compile and load custom op Just-In-Time.
# custom_relu_op_dup.cc is only used for multi ops test,
# not a new op, if you want to test only one op, remove this
# source file
source_files = ['custom_relu_op.cc']
if not IS_MAC:
    source_files.append('custom_relu_op.cu')

custom_module = load(
    name='custom_relu_for_model_jit',
    sources=source_files,
    extra_include_paths=paddle_includes,  # add for Coverage CI
    extra_cxx_cflags=extra_cc_args,  # test for cc flags
    extra_cuda_cflags=extra_nvcc_args,  # test for nvcc flags
    verbose=True)


class Net(nn.Layer):
    """
    A simple exmaple for Regression Model.
    """

    def __init__(self, in_dim, out_dim, use_custom_op=False):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, out_dim)
        self.relu_act = custom_module.custom_relu if use_custom_op else nn.functional.relu

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu_act(out)
        out = self.fc2(out)
        out = self.relu_act(out)

        out = paddle.mean(out, axis=-1)

        return out


class TestDygraphModel(unittest.TestCase):
    def setUp(self):

        self.seed = 2021
        self.in_dim = 10
        self.out_dim = 64
        self.batch_num = 10
        self.batch_size = 4
        self.datas = [
            np.random.uniform(
                size=[self.batch_size, self.in_dim]).astype('float32')
            for i in range(self.batch_num)
        ]
        self.labels = [
            np.random.uniform(size=[self.batch_size, 1]).astype('float32')
            for i in range(self.batch_num)
        ]

        self.devices = ['cpu', 'gpu'] if not IS_MAC else ['cpu']

        # for saving model
        self.model_path_template = "infer_model/custom_relu_dygaph_model_{}.pdparams"
        self.model_dy2stat_path = "infer_model/custom_relu_model_dy2sta"

        # for dy2stat
        self.x_spec = paddle.static.InputSpec(
            shape=[None, self.in_dim], dtype='float32', name='x')

    def func_train_eval(self):
        for device in self.devices:
            # set device
            paddle.set_device(device)

            # for train
            origin_relu_train_out = self.train_model(use_custom_op=False)
            custom_relu_train_out = self.train_model(use_custom_op=True)
            # open this when dy2stat is ready for eager 
            if _in_legacy_dygraph():
                custom_relu_dy2stat_train_out = self.train_model(
                    use_custom_op=True, dy2stat=True)  # for to_static
                self.assertTrue(
                    np.array_equal(origin_relu_train_out,
                                   custom_relu_dy2stat_train_out))

            self.assertTrue(
                np.array_equal(origin_relu_train_out, custom_relu_train_out))

            # for eval
            origin_relu_eval_out = self.eval_model(use_custom_op=False)
            custom_relu_eval_out = self.eval_model(use_custom_op=True)
            if _in_legacy_dygraph():
                custom_relu_dy2stat_eval_out = self.eval_model(
                    use_custom_op=True, dy2stat=True)  # for to_static
                self.assertTrue(
                    np.array_equal(origin_relu_eval_out,
                                   custom_relu_dy2stat_eval_out))

            self.assertTrue(
                np.array_equal(origin_relu_eval_out, custom_relu_eval_out))

    def test_train_eval(self):
        with _test_eager_guard():
            self.func_train_eval()
        self.func_train_eval()

    def train_model(self, use_custom_op=False, dy2stat=False):
        # reset random seed
        paddle.seed(self.seed)
        np.random.seed(self.seed)
        # paddle.framework.random._manual_program_seed(SEED)

        net = Net(self.in_dim, self.out_dim, use_custom_op)
        if dy2stat:
            net = paddle.jit.to_static(net, input_spec=[self.x_spec])
        mse_loss = paddle.nn.MSELoss()
        sgd = paddle.optimizer.SGD(learning_rate=0.1,
                                   parameters=net.parameters())

        for batch_id in range(self.batch_num):
            x = paddle.to_tensor(self.datas[batch_id])
            y = paddle.to_tensor(self.labels[batch_id])

            out = net(x)
            loss = mse_loss(out, y)

            loss.backward()
            sgd.minimize(loss)
            net.clear_gradients()

        # save inference model
        net.eval()
        if dy2stat:
            paddle.jit.save(net, self.model_dy2stat_path)
        else:
            paddle.save(net.state_dict(),
                        self.model_path_template.format(use_custom_op))

        return out.numpy()

    def eval_model(self, use_custom_op=False, dy2stat=False):
        net = Net(self.in_dim, self.out_dim, use_custom_op)

        if dy2stat:
            net = paddle.jit.load(self.model_dy2stat_path)
        else:
            state_dict = paddle.load(
                self.model_path_template.format(use_custom_op))
            net.set_state_dict(state_dict)

        sample_x = paddle.to_tensor(self.datas[0])
        net.eval()
        out = net(sample_x)

        return out.numpy()


class TestStaticModel(unittest.TestCase):
    def setUp(self):
        self.seed = 2021
        self.in_dim = 10
        self.out_dim = 64
        self.batch_num = 10
        self.batch_size = 8
        self.datas = [
            np.random.uniform(
                size=[self.batch_size, self.in_dim]).astype('float32')
            for i in range(self.batch_num)
        ]
        self.labels = [
            np.random.uniform(size=[self.batch_size, 1]).astype('float32')
            for i in range(self.batch_num)
        ]

        self.devices = ['cpu', 'gpu'] if not IS_MAC else ['cpu']

        # for saving model
        self.model_path_template = "infer_model/custom_relu_static_model_{}_{}"

        paddle.enable_static()

    def tearDown(self):
        paddle.disable_static()

    def test_train_eval(self):
        for device in self.devices:
            # for train
            original_relu_train_out = self.train_model(
                device, use_custom_op=False)
            custom_relu_train_out = self.train_model(device, use_custom_op=True)
            # using PE
            original_relu_train_pe_out = self.train_model(
                device, use_custom_op=False, use_pe=True)
            custom_relu_train_pe_out = self.train_model(
                device, use_custom_op=True, use_pe=True)

            self.assertTrue(
                np.array_equal(original_relu_train_out, custom_relu_train_out))
            self.assertTrue(
                np.array_equal(original_relu_train_pe_out,
                               custom_relu_train_pe_out))

            # for eval
            original_relu_eval_out = self.eval_model(
                device, use_custom_op=False)
            custom_relu_eval_out = self.eval_model(device, use_custom_op=True)
            # using PE
            original_relu_eval_pe_out = self.eval_model(
                device, use_custom_op=False, use_pe=True)
            custom_relu_eval_pe_out = self.eval_model(
                device, use_custom_op=True, use_pe=True)

            self.assertTrue(
                np.array_equal(original_relu_eval_out, custom_relu_eval_out))
            self.assertTrue(
                np.array_equal(original_relu_eval_pe_out,
                               custom_relu_eval_pe_out))

    def train_model(self, device, use_custom_op=False, use_pe=False):
        # reset random seed
        paddle.seed(self.seed)
        np.random.seed(self.seed)
        # set device
        paddle.set_device(device)

        with paddle.static.scope_guard(paddle.static.Scope()):
            with paddle.static.program_guard(paddle.static.Program(),
                                             paddle.static.Program()):
                x = paddle.static.data(
                    shape=[None, self.in_dim], name='x', dtype='float32')
                y = paddle.static.data(
                    shape=[None, 1], name='y', dtype='float32')

                net = Net(self.in_dim, self.out_dim, use_custom_op)
                out = net(x)

                loss = nn.functional.mse_loss(out, y)
                sgd = paddle.optimizer.SGD(learning_rate=0.01)
                sgd.minimize(loss)

                exe = exe = paddle.static.Executor()
                exe.run(paddle.static.default_startup_program())

                # For PE
                if use_pe:
                    places = paddle.static.cpu_places(
                    ) if device is 'cpu' else paddle.static.cuda_places()
                    main_program = paddle.static.CompiledProgram(
                        paddle.static.default_main_program(
                        )).with_data_parallel(
                            loss_name=loss.name, places=places)
                else:
                    main_program = paddle.static.default_main_program()

                for batch_id in range(self.batch_num):
                    x_data = self.datas[batch_id]
                    y_data = self.labels[batch_id]

                    res = exe.run(main_program,
                                  feed={'x': x_data,
                                        'y': y_data},
                                  fetch_list=[out])

                # save model
                paddle.static.save_inference_model(
                    self.model_path_template.format(use_custom_op, use_pe),
                    [x], [out], exe)

                return res[0]

    def eval_model(self, device, use_custom_op=False, use_pe=False):
        paddle.set_device(device)

        with paddle.static.scope_guard(paddle.static.Scope()):
            with paddle.static.program_guard(paddle.static.Program()):
                exe = paddle.static.Executor()

                [inference_program, feed_target_names,
                 fetch_targets] = paddle.static.load_inference_model(
                     self.model_path_template.format(use_custom_op, use_pe),
                     exe)

                x_data = self.datas[0]
                results = exe.run(inference_program,
                                  feed={feed_target_names[0]: x_data},
                                  fetch_list=fetch_targets)

                return results[0]


if __name__ == '__main__':
    unittest.main()
