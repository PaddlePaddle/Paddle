#   copyright (c) 2019 paddlepaddle authors. all rights reserved.
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
import six
import paddle
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import QuantizationFreezePass
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.fluid.contrib.slim.quantization import TransformForMkldnnPass 
from paddle.fluid import core

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CPU_NUM"] = "1"


class TestMKLDNNTransformBasedFreezePass(unittest.TestCase):
    def freeze_graph(self,
                     use_cuda,
                     seed,
                     activation_quant_type,
                     weight_quant_type='abs_max',
                     for_ci=False):
        def build_program(main, startup, is_test):
            main.random_seed = seed
            startup.random_seed = seed
            with fluid.unique_name.guard():
                with fluid.program_guard(main, startup):
                    img = fluid.layers.data(
                        name='image', shape=[1, 28, 28], dtype='float32')
                    label = fluid.layers.data(
                        name='label', shape=[1], dtype='int64')
                    loss = conv_net(img, label)
                    if not is_test:
                        opt = fluid.optimizer.Adam(learning_rate=0.001)
                        opt.minimize(loss)
            return [img, label], loss
            
        def check_program(self, transform_pass, program):
            for block in program.blocks:
                for op in block.ops:
                    if op.type in self.quantizable_op_and_inputs:
                        for arg_name in op.output_arg_names:
                            # Check quantizable op's output is linked to 
                            # fake_dequantize's output
                            self.assertTrue(
                                arg_name.endswith('.dequantized'))


        random.seed(0)
        np.random.seed(0)

        main = fluid.Program()
        startup = fluid.Program()
        test_program = fluid.Program()
        feeds, loss = build_program(main, startup, False)
        build_program(test_program, startup, True)
        test_program = test_program.clone(for_test=True)
        main_graph = IrGraph(core.Graph(main.desc), for_test=False)
        test_graph = IrGraph(core.Graph(test_program.desc), for_test=True)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        scope = fluid.Scope()
        with fluid.scope_guard(scope):
            exe.run(startup)
        transform_pass = QuantizationTransformPass(
            scope=scope,
            place=place,
            activation_quantize_type=activation_quant_type,
            weight_quantize_type=weight_quant_type)
        transform_pass.apply(main_graph)
        transform_pass.apply(test_graph)

        build_strategy = fluid.BuildStrategy()
        build_strategy.memory_optimize = False
        build_strategy.enable_inplace = False
        binary = fluid.CompiledProgram(main_graph.graph).with_data_parallel(
            loss_name=loss.name, build_strategy=build_strategy)
        quantized_test_program = test_graph.to_program()
        iters = 5
        batch_size = 8

        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=500),
            batch_size=batch_size)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=batch_size)
        feeder = fluid.DataFeeder(feed_list=feeds, place=place)
        with fluid.scope_guard(scope):
            for _ in range(iters):
                data = next(train_reader())
                loss_v = exe.run(binary,
                                 feed=feeder.feed(data),
                                 fetch_list=[loss])


        # Freeze graph for inference, but the weight of fc/conv is still float type.
        freeze_pass = QuantizationFreezePass(
            scope=scope, place=place, weight_quantize_type=weight_quant_type)
        freeze_pass.apply(test_graph)

       # Transform quantized graph for MKL-DNN INT8 inference
        mkldnn_int8_pass = TransformForMkldnnPass(
            scope=scope, place=place)
        mkldnn_int8_pass.apply(test_graph)
        if not for_ci:
            marked_nodes = set()
            for op in test_graph.all_op_nodes():
                if op.name().find('quantize') > -1:
                    marked_nodes.add(op)
            test_graph.draw('.', 'test_freeze' + dev_name +
                            activation_quant_type + '_' + weight_quant_type,
                            marked_nodes)
        server_program = test_graph.to_program()
        w_mkldnn = np.array(scope.find_var('conv2d_1.w_0').get_tensor())
        self.assertFalse(w_mkldnn.is_integer())
        check_program(server_program)
        dev_name = '_gpu_' if use_cuda else '_cpu_'
        if not for_ci:
            print('{}: {}'.format('w_mkldnn' + dev_name + activation_quant_type +
                                  '_' + weight_quant_type, np.sum(w_mkldnn)))
        


def test_freeze_graph_cpu_static(self):
        with fluid.unique_name.guard():
            self.freeze_graph(
                False,
                seed=2,
                activation_quant_type='range_abs_max',
                weight_quant_type='abs_max',
                for_ci=True)
            self.freeze_graph(
                False,
                seed=2,
                activation_quant_type='moving_average_abs_max',
                weight_quant_type='abs_max',
                for_ci=True)



if __name__ == '__main__':
    unittest.main()
