#   copyright (c) 2018 paddlepaddle authors. all rights reserved.
#
# licensed under the apache license, version 2.0 (the "license");
# you may not use this file except in compliance with the license.
# you may obtain a copy of the license at
#
#     http://www.apache.org/licenses/license-2.0
#
# unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "as is" basis,
# without warranties or conditions of any kind, either express or implied.
# see the license for the specific language governing permissions and
# limitations under the license.

import os
import unittest
import random
import numpy as np
import paddle.fluid as fluid
import paddle
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization.quantize_transpiler_v2 import (
    QuantizeTranspilerV2,
)
from paddle.fluid import core

paddle.enable_static()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CPU_NUM"] = "1"


def conv_net(img, label):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        pool_type='max',
        act="relu",
    )
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        pool_type='avg',
        act="relu",
    )
    with fluid.name_scope("skip_quant"):
        hidden = fluid.layers.fc(input=conv_pool_1, size=100, act='relu')
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = paddle.nn.functional.cross_entropy(
        input=prediction, label=label, reduction='none', use_softmax=False
    )
    avg_loss = paddle.mean(loss)
    return avg_loss


class TestQuantizeProgramPass(unittest.TestCase):
    def quantize_program(
        self,
        use_cuda,
        seed,
        activation_quant_type='abs_max',
        weight_quant_type='abs_max',
        for_ci=False,
    ):
        def build_program(main, startup, is_test):
            main.random_seed = seed
            startup.random_seed = seed
            with fluid.unique_name.guard():
                with fluid.program_guard(main, startup):
                    img = fluid.layers.data(
                        name='image', shape=[1, 28, 28], dtype='float32'
                    )
                    label = fluid.layers.data(
                        name='label', shape=[1], dtype='int64'
                    )
                    loss = conv_net(img, label)
                    if not is_test:
                        opt = fluid.optimizer.Adam(learning_rate=0.0001)
                        opt.minimize(loss)
            return [img, label], loss

        random.seed(0)
        np.random.seed(0)

        # 1 Define program
        train_program = fluid.Program()
        startup_program = fluid.Program()
        test_program = fluid.Program()
        feeds, loss = build_program(train_program, startup_program, False)
        build_program(test_program, startup_program, True)
        test_program = test_program.clone(for_test=True)

        if not for_ci:
            train_graph = IrGraph(
                core.Graph(train_program.desc), for_test=False
            )
            train_graph.draw('.', 'train_program_1')
            test_graph = IrGraph(core.Graph(test_program.desc), for_test=True)
            test_graph.draw('.', 'test_program_1')

        # 2 Apply quantization
        qt = QuantizeTranspilerV2(
            activation_quantize_type=activation_quant_type,
            weight_quantize_type=weight_quant_type,
        )
        qt.apply(train_program, startup_program, is_test=False)
        qt.apply(test_program, startup_program, is_test=True)

        # 3 Train
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        exe = fluid.Executor(place)
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup_program)
        if not for_ci:
            train_graph = IrGraph(
                core.Graph(train_program.desc), for_test=False
            )
            train_graph.draw('.', 'train_program_2')
            test_graph = IrGraph(core.Graph(test_program.desc), for_test=True)
            test_graph.draw('.', 'test_program_2')

        build_strategy = fluid.BuildStrategy()
        build_strategy.memory_optimize = False
        build_strategy.enable_inplace = False
        build_strategy.fuse_all_reduce_ops = False
        binary = fluid.CompiledProgram(train_program).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy
        )
        iters = 5
        batch_size = 8

        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=batch_size
        )
        feeder = fluid.DataFeeder(feed_list=feeds, place=place)
        with fluid.scope_guard(scope):
            for idx in range(iters):
                data = next(train_reader())
                loss_v = exe.run(
                    binary, feed=feeder.feed(data), fetch_list=[loss]
                )
                if not for_ci and idx % 20 == 0:
                    print('{}: {}'.format('loss', np.mean(loss_v)))

        print('{}: {}'.format('loss', np.mean(loss_v)))

        # 4 Convert
        qt.convert(test_program, scope)
        if not for_ci:
            with fluid.scope_guard(scope):
                fluid.io.save_inference_model(
                    './infer_model',
                    ['image', 'label'],
                    [loss],
                    exe,
                    test_program,
                    clip_extra=True,
                )

    def test_gpu_1(self):
        if fluid.core.is_compiled_with_cuda():
            self.quantize_program(
                use_cuda=True,
                seed=1,
                activation_quant_type='abs_max',
                weight_quant_type='abs_max',
                for_ci=True,
            )

    def test_gpu_2(self):
        if fluid.core.is_compiled_with_cuda():
            self.quantize_program(
                use_cuda=True,
                seed=1,
                activation_quant_type='moving_average_abs_max',
                weight_quant_type='channel_wise_abs_max',
                for_ci=True,
            )

    def test_cpu_1(self):
        self.quantize_program(
            use_cuda=False,
            seed=2,
            activation_quant_type='abs_max',
            weight_quant_type='abs_max',
            for_ci=True,
        )

    def test_cpu_2(self):
        self.quantize_program(
            use_cuda=False,
            seed=2,
            activation_quant_type='moving_average_abs_max',
            weight_quant_type='channel_wise_abs_max',
            for_ci=True,
        )


if __name__ == '__main__':
    unittest.main()
