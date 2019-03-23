# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import logging
import sys
import numpy as np
from .... import Executor
from .... import io
from .... import core
from ....compiler import CompiledProgram
from ....framework import IrGraph
from ..core.strategy import Strategy
from quantization_pass import *

__all__ = ['QuantizationStrategy']

logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s')
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class QuantizationStrategy(Strategy):
    """
    The strategy for Quantization.
    """

    def __init__(self,
                 start_epoch=0,
                 end_epoch=0,
                 float_model_save_path=None,
                 mobile_model_save_path=None,
                 int8_model_save_path=None,
                 activation_bits=8,
                 weight_bits=8,
                 activation_quantize_type='abs_max',
                 save_in_nodes=None,
                 save_out_nodes=None):
        """
        Args:
            start_epoch(int): The 'on_epoch_begin' function will be called in start_epoch. default: 0
            end_epoch(int): The 'on_epoch_end' function will be called in end_epoch. default: 0
            float_model_save_path(str): The path to save model with float weights. 
                            None means it doesn't save float model. defalut: None.
            mobile_model_save_path(str): The path to save model for paddle-mobile execution.
                            None means it doesn't save mobile model. defalut: None.
            int8_model_save_path(str): The path to save model with int8_t weight.
                            None means it doesn't save int8 model. defalut: None.
            activation_bits(int): quantization bit number for activation. default: 8.
            weight_bits(int): quantization bit number for weights. The bias is not quantized.
                              default: 8.
            activation_quantize_type(str): quantization type for activation,
                now support 'abs_max', 'range_abs_max' and 'moving_average_abs_max'.
                If use 'abs_max' mode, the quantization scale will be calculated
                dynamically each step in both training and testing period. If use
                'range_abs_max', a static quantization scale will be calculated
                during training and used in inference.
            save_in_nodes(list<str>): A list of variable names used to prune graph 
                                      for saving inference model.
            save_out_nodes(list<str>): A list of variable names used to prune graph 
                                      for saving inference model.

        """
        super(QuantizationStrategy, self).__init__(start_epoch, end_epoch)
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.float_model_save_path = float_model_save_path
        self.mobile_model_save_path = mobile_model_save_path
        self.int8_model_save_path = int8_model_save_path
        self.activation_bits = activation_bits
        self.weight_bits = weight_bits
        self.activation_quantize_type = activation_quantize_type
        self.save_out_nodes = save_out_nodes
        self.save_in_nodes = save_in_nodes

    def on_epoch_begin(self, context):
        """
        Insert fake_quantize_op and fake_dequantize_op before trainging and testing.
        """
        super(QuantizationStrategy, self).on_compression_begin(context)
        if self.start_epoch == context.epoch_id:
            _logger.info('QuantizationStrategy::on_epoch_begin')
            train_ir_graph = IrGraph(
                core.Graph(context.optimize_graph.program.desc), for_test=False)
            test_ir_graph = IrGraph(
                core.Graph(context.eval_graph.program.desc), for_test=True)
            transform_pass = QuantizationTransformPass(
                scope=context.scope,
                place=context.place,
                weight_bits=self.weight_bits,
                activation_bits=self.activation_bits,
                activation_quantize_type=self.activation_quantize_type)
            transform_pass.apply(train_ir_graph)
            transform_pass.apply(test_ir_graph)

            # for quantization training
            context.optimize_graph.compiled_graph = CompiledProgram(
                train_ir_graph.graph).with_data_parallel(
                    loss_name=context.optimize_graph.out_nodes['loss'])
            # for evaluation. And program compiled from ir graph must be with data parallel.
            context.eval_graph.compiled_graph = CompiledProgram(
                test_ir_graph.graph).with_data_parallel()
            # for saving inference model after training
            context.put('quantization_test_ir_graph_backup', test_ir_graph)
            _logger.info('Finish QuantizationStrategy::on_epoch_begin')

    def on_epoch_end(self, context):
        """
        Free and save inference model.
        """
        super(QuantizationStrategy, self).on_compression_end(context)

        if context.epoch_id == self.end_epoch:
            _logger.info('QuantizationStrategy::on_epoch_end')
            test_ir_graph = context.get('quantization_test_ir_graph_backup')
            # freeze the graph after training
            freeze_pass = QuantizationFreezePass(
                scope=context.scope,
                place=context.place,
                weight_bits=self.weight_bits,
                activation_bits=self.activation_bits)
            freeze_pass.apply(test_ir_graph)

            # for other strategies
            context.eval_graph.program = test_ir_graph.to_program()

            if self.save_out_nodes == None:
                out_vars = [
                    context.eval_graph.get_var(var_name)
                    for var_name in context.eval_graph.out_nodes.values()
                ]
            else:
                out_vars = [
                    context.eval_graph.get_var(var_name)
                    for var_name in self.save_out_nodes
                ]

            if self.save_in_nodes == None:
                in_vars = list(context.eval_graph.out_nodes.values())
            else:
                in_vars = self.save_in_nodes

            # save float model
            if self.float_model_save_path:
                executor = Executor(context.place)
                io.save_inference_model(
                    self.float_model_save_path,
                    in_vars,
                    out_vars,
                    executor,
                    main_program=test_ir_graph.to_program(),
                    model_filename='model',
                    params_filename='weights',
                    export_for_deployment=True)

            # save int8 model
            if self.int8_model_save_path:
                convert_int8_pass = ConvertToInt8Pass(
                    scope=context.scope, place=context.place)
                convert_int8_pass.apply(test_ir_graph)
                executor = Executor(context.place)
                io.save_inference_model(
                    self.int8_model_save_path,
                    in_vars,
                    out_vars,
                    executor,
                    main_program=test_ir_graph.to_program(),
                    model_filename='model',
                    params_filename='weights',
                    export_for_deployment=True)

            # save mobile model
            if self.mobile_model_save_path:
                if not self.int8_model_save_path:
                    # convert the weights as int8_t type
                    convert_int8_pass = ConvertToInt8Pass(
                        scope=context.scope, place=context.place)
                    convert_int8_pass.apply(test_ir_graph)
                # make some changes on the graph for the mobile inference
                mobile_pass = TransformForMobilePass()
                mobile_pass.apply(test_ir_graph)
                executor = Executor(context.place)
                io.save_inference_model(
                    self.mobile_model_save_path,
                    in_vars,
                    out_vars,
                    executor,
                    main_program=test_ir_graph.to_program(),
                    model_filename='model',
                    params_filename='weights',
                    export_for_deployment=True)
            _logger.info('Finish QuantizationStrategy::on_epoch_end')
