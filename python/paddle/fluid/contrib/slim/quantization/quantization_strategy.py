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

import numpy as np
from .... import Executor
from .... import io
from ..core.strategy import Strategy

__all__ = ['QuantizationStrategy']


class QuantizationStrategy(Strategy):
    """
    The strategy for Quantization.
    """

    def __init__(self,
                 quantizer,
                 start_epoch=0,
                 end_epoch=10,
                 dirname=None,
                 target_device='mobile',
                 save_as_int8=True,
                 activation_quantize_type='abs_max'):
        super(QuantizationStrategy, self).__init__(start_epoch, end_epoch)
        self.quantizer = quantizer
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.target_device = target_device
        self.save_as_int8 = save_as_int8
        self.activation_quantize_type = activation_quantize_type
        self.test_graph = None

    def on_epoch_begin(self, context):
        super(QuantizationStrategy, self).on_compression_begin(context)
        if self.start_epoch == context.epoch_id:

            train_graph = IrGraph(
                core.Graph(context.optimize_graph.program.desc), for_test=False)
            self.test_graph = IrGraph(
                core.Graph(context.eval_graph.program.desc), for_test=True)
            # insert fake_quantize_op and fake_dequantize_op before trainging and testing
            transform_pass = QuantizationTransformPass(
                scope=context.optimize_graph.scope,
                place=context.palce,
                activation_quantize_type=self.activation_quantize_type)
            transform_pass.apply(train_graph)
            transform_pass.apply(test_graph)
            binary = fluid.CompiledProgram(
                train_graph.graph).with_data_parallel(
                    loss_name=context.optimize_graph.out_nodes['cost'])
            context.eval_graph.program = test_graph.to_program()

            context.optimize_graph.program = binary

    def on_epoch_end(self, context):
        super(QuantizationStrategy, self).on_compression_end(context)

        if context.epoch_id == self.end_epoch:
            scope = context.eval_graph.scope
            # freeze the graph after training
            freeze_pass = QuantizationFreezePass(
                scope=scope, place=context.place)
            freeze_pass.apply(self.test_graph)
            context.eval_graph.program = self.test_graph.to_program()

            if self.int8_model_save_path:
                # convert the weights as int8_t type
                convert_int8_pass = ConvertToInt8Pass(
                    scope=scope, place=context.place)
                convert_int8_pass.apply(self.test_graph)
                io.save_params(
                    main_program=self.test_graph.to_program(),
                    scope=scope, )

                executor = Executor(self.place)
                io.save_inference_model(
                    self.int8_model_save_path,
                    context.eval_graph.in_nodes.keys(), [
                        context.eval_graph.get_var(var_name)
                        for var_name in context.eval_graph.out_nodes.values()
                    ],
                    executor,
                    main_program=self.test_graph.to_program(),
                    model_filename='model',
                    params_filename='weights')

            if self.mobile_model_save_path:
                if not self.int8_model_save_path:
                    # convert the weights as int8_t type
                    convert_int8_pass = ConvertToInt8Pass(
                        scope=scope, place=context.place)
                    convert_int8_pass.apply(self.test_graph)
                # make some changes on the graph for the mobile inference
                mobile_pass = TransformForMobilePass()
                mobile_pass.apply(self.test_graph)

                executor = Executor(self.place)
                io.save_inference_model(
                    self.mobile_model_save_path,
                    context.eval_graph.in_nodes.keys(), [
                        context.eval_graph.get_var(var_name)
                        for var_name in context.eval_graph.out_nodes.values()
                    ],
                    executor,
                    main_program=self.test_graph.to_program(),
                    model_filename='model',
                    params_filename='weights')
