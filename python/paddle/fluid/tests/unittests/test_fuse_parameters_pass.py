# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest

import numpy as np
import paddle.fluid.core as core
import os
import paddle.fluid as fluid

from paddle.fluid import compiler


def simple_fc_net():
    img = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    output = []
    hidden = img
    for _ in range(4):
        hidden = fluid.layers.fc(
            hidden,
            size=200,
            act='tanh',
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Constant(value=1.0)))
        output.append(hidden)
    prediction = fluid.layers.fc(hidden, size=10, act='softmax')
    output.append(prediction)
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    output.append(loss)
    loss = fluid.layers.mean(loss)
    return [loss] + output


def fc_with_batchnorm():
    img = fluid.layers.data(name='image', shape=[784], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    output = []
    hidden = img
    for _ in range(1):
        with fluid.name_scope("hidden"):
            hidden = fluid.layers.fc(
                hidden,
                size=200,
                act='tanh',
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.Constant(value=1.0)))
            output.append(hidden)
            hidden = fluid.layers.batch_norm(input=hidden)
            output.append(hidden)
    with fluid.name_scope("fc_layer"):
        prediction = fluid.layers.fc(hidden, size=10, act='softmax')
        output.append(prediction)
    with fluid.name_scope("loss"):
        loss = fluid.layers.cross_entropy(input=prediction, label=label)
        output.append(loss)
        loss = fluid.layers.mean(loss)
    return [loss] + output


def init_data():
    np.random.seed(5)
    img = np.random.random(size=[32, 784]).astype(np.float32)
    label = np.ones(shape=[32, 1], dtype='int64')
    return img, label


class TestSimpleNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ['CPU_NUM'] = str(4)

    def run_with_parallel_executor(self,
                                   fuse_parameter,
                                   feed,
                                   main,
                                   output,
                                   startup,
                                   use_cuda,
                                   loss_name=None):
        fetch_list = [out.name for out in output]
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup)
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = 1
        build_strategy = fluid.BuildStrategy()
        build_strategy.enable_inplace = False
        build_strategy.fuse_parameters_pass = fuse_parameter

        mult_graph = compiler.CompiledProgram(main).with_data_parallel(
            loss_name=loss_name if loss_name is not None else "",
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

        res = exe.run(mult_graph, feed=feed, fetch_list=fetch_list)

        return res

    def run_model(self,
                  model,
                  use_cuda,
                  feed,
                  fuse_parameter=False,
                  optimizer=None):
        main = fluid.Program()
        startup = fluid.Program()
        startup.random_seed = 1  # Fix random seed
        main.random_seed = 1
        with fluid.program_guard(main, startup):
            output = model()
            if optimizer:
                optimizer().minimize(output[0])

        res = self.run_with_parallel_executor(
            fuse_parameter,
            feed,
            main,
            output,
            startup,
            use_cuda,
            loss_name=output[0].name if optimizer else None)
        return res

    def check_result(self, model, opt_alg=fluid.optimizer.Adam):
        img, label = init_data()

        def _run_with_fuse_parameter(fuse_parameter):
            return self.run_model(
                model,
                use_cuda,
                feed={"image": img,
                      "label": label},
                fuse_parameter=fuse_parameter,
                optimizer=opt_alg)

        for use_cuda in [True, False]:
            if use_cuda and not core.is_compiled_with_cuda():
                continue
            output = _run_with_fuse_parameter(False)
            output2 = _run_with_fuse_parameter(True)

            for output in zip(output, output2):
                if not (output[0] == output[1]).all():
                    print(output[0])
                    print(output[1])
                    assert (
                        output[0] == output[1]).all(), "There have some diff "

    def test_with_adam_optimize(self):
        self.check_result(simple_fc_net)
        self.check_result(fc_with_batchnorm)

    def test_with_sgd_optimize(self):
        def _sgd_optimizer(learning_rate=1e-2):
            return fluid.optimizer.SGD(learning_rate=learning_rate)

        self.check_result(simple_fc_net, opt_alg=_sgd_optimizer)
        self.check_result(fc_with_batchnorm, opt_alg=_sgd_optimizer)


if __name__ == '__main__':
    unittest.main()
