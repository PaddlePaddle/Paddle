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

import numpy as np
import unittest
import itertools
import abc
import logging
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.inference as paddle_infer

from paddle import compat as cpt
from typing import *
from program_config import TensorConfig, OpConfig, ProgramConfig
from auto_scan_test import AutoScanTest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(filename)s - %(message)s")


class TrtLayerAutoScanTest(AutoScanTest):
    class TensorRTParam:
        '''
        TensorRT subgraph engine parameters. 
        '''

        def __init__(self, workspace_size, max_batch_size, min_subgraph_size,
                     precision, use_static, use_calib_mode):
            self.workspace_size = workspace_size
            self.max_batch_size = max_batch_size
            self.min_subgraph_size = min_subgraph_size
            self.precision = precision
            self.use_static = use_static
            self.use_calib_mode = use_calib_mode

    class DynamicShapeParam:
        '''
         Prepare TensorRT subgraph engine dynamic shape parameters. 
         '''

        def __init__(self, min_input_shape, max_input_shape, optim_input_shape,
                     disable_trt_plugin_fp16):
            self.min_input_shape = min_input_shape
            self.max_input_shape = max_input_shape
            self.optim_input_shape = optim_input_shape
            self.disable_trt_plugin_fp16 = disable_trt_plugin_fp16

    def __init__(self, methodName='runTest'):
        super(TrtLayerAutoScanTest, self).__init__(methodName)
        self.trt_param = self.TensorRTParam(
            workspace_size=0,
            max_batch_size=4,
            min_subgraph_size=0,
            precision=paddle_infer.PrecisionType.Float32,
            use_static=False,
            use_calib_mode=False)
        self.dynamic_shape = self.DynamicShapeParam({}, {}, {}, False)

    def update_program_input_and_weight_with_attr(self, op_attr_list):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_program_configs(self):
        all_op_attrs_keys = []
        all_op_attrs_values = []
        for op_config in self.ops_config:
            all_op_attrs_keys.append(list(op_config["op_attrs"].keys()))
            all_op_attrs_values.extend(list(op_config["op_attrs"].values()))
        if len(all_op_attrs_values) == 0:
            all_op_attrs_values.append([None])
        for attrs_sample in itertools.product(*all_op_attrs_values):
            op_attr_list = []
            index = 0
            ops = []
            for op_config in self.ops_config:
                op_attr = dict(
                    zip(
                        list(op_config["op_attrs"].keys()), attrs_sample[
                            index:index + len(op_config["op_attrs"])]))
                op_attr_list.append(op_attr)
                index = index + len(op_config["op_attrs"])
                ops.append(
                    OpConfig(
                        type=op_config["op_type"],
                        inputs=op_config["op_inputs"],
                        outputs=op_config["op_outputs"],
                        attrs=op_attr))

            self.update_program_input_and_weight_with_attr(op_attr_list)
            program_config = ProgramConfig(
                ops=ops,
                weights=self.program_weights,
                inputs=self.program_inputs,
                outputs=self.program_outputs)
            yield program_config

    def create_program_config(
            self, use_trt=True,
            precision_mode=paddle_infer.PrecisionType.Float32):
        config = paddle_infer.Config()
        config.enable_use_gpu(100, 0)
        if use_trt:
            config.switch_ir_debug()
            config.enable_tensorrt_engine(
                max_batch_size=self.trt_param.max_batch_size,
                workspace_size=self.trt_param.workspace_size,
                min_subgraph_size=self.trt_param.min_subgraph_size,
                precision_mode=precision_mode,
                use_static=self.trt_param.use_static,
                use_calib_mode=self.trt_param.use_calib_mode)
            if len(self.dynamic_shape.min_input_shape
                   ) != 0 and self.dynamic_shape.min_input_shape.keys(
                   ) == self.dynamic_shape.max_input_shape.keys(
                   ) and self.dynamic_shape.min_input_shape.keys(
                   ) == self.dynamic_shape.opt_input_shape.keys():
                config.set_trt_dynamic_shape_info(
                    self.dynamic_shape.min_input_shape,
                    self.dynamic_shape.max_input_shape,
                    self.dynamic_shape.opt_input_shape,
                    self.dynamic_shape.disable_trt_plugin_fp16)
        return config

    @abc.abstractmethod
    def sample_predictor_configs(self):
        logging.info('--------- gpu inference ---------')
        yield self.create_program_config(use_trt=False)
        logging.info('--------- trt inference ---------')
        yield self.create_program_config(
            use_trt=True, precision_mode=self.trt_param.precision)
