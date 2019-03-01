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
from .... import core
from ....compiler import CompiledProgram
from ....framework import IrGraph
from quantization_pass import *
from ..core.strategy import Strategy
import logging
import sys

__all__ = ['QuantizationStrategy']

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


class QuantizationStrategy(Strategy):
    """
    The strategy for Quantization.
    """

    def __init__(self,
                 start_epoch=0,
                 end_epoch=10,
                 float_model_save_path=None,
                 mobile_model_save_path=None,
                 int8_model_save_path=None,
                 activation_quantize_type='abs_max'):
        super(QuantizationStrategy, self).__init__(start_epoch, end_epoch)
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.float_model_save_path = float_model_save_path
        self.mobile_model_save_path = mobile_model_save_path
        self.int8_model_save_path = int8_model_save_path
        self.activation_quantize_type = activation_quantize_type
        self.test_graph = None

    def on_epoch_begin(self, context):
        super(QuantizationStrategy, self).on_compression_begin(context)
        if self.start_epoch == context.epoch_id:
            logger.info('QuantizationStrategy::on_epoch_begin')
            train_graph = IrGraph(
                core.Graph(context.optimize_graph.program.desc), for_test=False)
            self.test_graph = IrGraph(
                core.Graph(context.eval_graph.program.desc), for_test=True)
            # insert fake_quantize_op and fake_dequantize_op before trainging and testing
            transform_pass = QuantizationTransformPass(
                scope=context.optimize_graph.scope,
                place=context.place,
                activation_quantize_type=self.activation_quantize_type)
            transform_pass.apply(train_graph)
            transform_pass.apply(self.test_graph)
            binary = CompiledProgram(train_graph.graph).with_data_parallel(
                loss_name=context.optimize_graph.out_nodes['cost'])

            context.optimize_graph.compiled_program = binary
            context.eval_graph.program = self.test_graph.to_program()

            logger.info('Finish QuantizationStrategy::on_epoch_begin')

    def on_epoch_end(self, context):
        super(QuantizationStrategy, self).on_compression_end(context)

        if context.epoch_id == self.end_epoch:
            logger.info('QuantizationStrategy::on_epoch_end')
            scope = context.eval_graph.scope
            # freeze the graph after training
            freeze_pass = QuantizationFreezePass(
                scope=scope, place=context.place)
            logger.info('create QuantizationFreezePass')
            freeze_pass.apply(self.test_graph)
            logger.info('apply QuantizationFreezePass')
            context.eval_graph.program = self.test_graph.to_program()
            logger.info('test_graph to_program')

            if self.float_model_save_path:
                executor = Executor(context.place)
                io.save_inference_model(
                    self.float_model_save_path,
                    context.eval_graph.in_nodes.keys(), [
                        context.eval_graph.get_var(var_name)
                        for var_name in context.eval_graph.out_nodes.values()
                    ],
                    executor,
                    main_program=self.test_graph.to_program(),
                    model_filename='model',
                    params_filename='weights',
                    export_for_deployment=False)

            if self.int8_model_save_path:
                # convert the weights as int8_t type
                convert_int8_pass = ConvertToInt8Pass(
                    scope=scope, place=context.place)
                convert_int8_pass.apply(self.test_graph)
                executor = Executor(context.place)
                io.save_inference_model(
                    self.int8_model_save_path,
                    context.eval_graph.in_nodes.keys(), [
                        context.eval_graph.get_var(var_name)
                        for var_name in context.eval_graph.out_nodes.values()
                    ],
                    executor,
                    main_program=self.test_graph.to_program(),
                    model_filename='model',
                    params_filename='weights',
                    export_for_deployment=False)

            if self.mobile_model_save_path:
                if not self.int8_model_save_path:
                    # convert the weights as int8_t type
                    convert_int8_pass = ConvertToInt8Pass(
                        scope=scope, place=context.place)
                    convert_int8_pass.apply(self.test_graph)
                # make some changes on the graph for the mobile inference
                mobile_pass = TransformForMobilePass()
                mobile_pass.apply(self.test_graph)
                executor = Executor(context.place)
                io.save_inference_model(
                    self.mobile_model_save_path,
                    context.eval_graph.in_nodes.keys(), [
                        context.eval_graph.get_var(var_name)
                        for var_name in context.eval_graph.out_nodes.values()
                    ],
                    executor,
                    main_program=self.test_graph.to_program(),
                    model_filename='model',
                    params_filename='weights',
                    export_for_deployment=False)
            logger.info('Finish QuantizationStrategy::on_epoch_end')
