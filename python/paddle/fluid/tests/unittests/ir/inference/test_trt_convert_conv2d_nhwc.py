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

from trt_layer_auto_scan_test import TrtLayerAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import numpy as np
import unittest
import itertools
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import time
from program_config import create_fake_model, create_quant_model
import os
import shutil


class TrtConvertConv2dTest(TrtLayerAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        inputs = program_config.inputs
        weights = program_config.weights
        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        if inputs['input_data'].shape[3] != weights['conv2d_weight'].shape[
                1] * attrs[0]['groups']:
            return False
        ver = paddle_infer.get_trt_compile_version()
        if ver[0] * 1000 + ver[1] * 100 + ver[0] * 10 < 7000:
            if attrs[0]['padding_algorithm'] == 'SAME' and (
                    attrs[0]['strides'][0] > 1 or attrs[0]['strides'][1] > 1):
                return False

        return True

    def sample_program_configs(self):
        self.trt_param.workspace_size = 1073741824

        def generate_input1(batch, attrs: List[Dict[str, Any]]):
            return np.ones(
                [batch, 64, 64, 3 * attrs[0]['groups']]).astype(np.float32)

        def generate_weight1(attrs: List[Dict[str, Any]]):
            return np.random.random([24, 3, 3, 3]).astype(np.float32)

        batch_list = [1, 4]
        strides_list = [[1, 1], [2, 2], [1, 2]]
        paddings_list = [[0, 3], [1, 2, 3, 4]]
        groups_list = [1, 3]
        padding_altorithm_list = ['EXPLICIT', 'SAME', 'VALID']
        dilations_list = [[1, 1], [2, 2], [1, 2]]
        data_format_list = ['NHWC']

        combination = [
            batch_list, strides_list, paddings_list, groups_list,
            padding_altorithm_list, dilations_list, data_format_list
        ]
        for batch, strides, paddings, groups, padding_algorithm, dilations, data_format in itertools.product(
                *combination):
            dics = [{
                "data_fromat": data_format,
                "dilations": dilations,
                "padding_algorithm": padding_algorithm,
                "groups": groups,
                "paddings": paddings,
                "strides": strides,
                "data_format": data_format
            }, {}]

            ops_config = [{
                "op_type": "conv2d",
                "op_inputs": {
                    "Input": ["input_data"],
                    "Filter": ["conv2d_weight"]
                },
                "op_outputs": {
                    "Output": ["conv_output_data"]
                },
                "op_attrs": dics[0]
            }, {
                "op_type": "relu",
                "op_inputs": {
                    "X": ["conv_output_data"]
                },
                "op_outputs": {
                    "Out": ["output_data"]
                },
                "op_attrs": dics[1]
            }]

            ops = self.generate_op_config(ops_config)

            program_config = ProgramConfig(
                ops=ops,
                weights={
                    "conv2d_weight":
                    TensorConfig(data_gen=partial(generate_weight1, dics))
                },
                inputs={
                    "input_data": TensorConfig(data_gen=partial(generate_input1,
                                                                batch, dics))
                },
                outputs=["output_data"])

            yield program_config

    def sample_predictor_configs(
            self, program_config) -> (paddle_infer.Config, List[int], float):
        def generate_dynamic_shape(attrs):
            self.dynamic_shape.min_input_shape = {
                "input_data": [1, 32, 32, 3 * attrs[0]['groups']],
                "output_data": [1, 32, 32, 24]
            }
            self.dynamic_shape.max_input_shape = {
                "input_data": [4, 64, 64, 3 * attrs[0]['groups']],
                "output_data": [4, 64, 64, 24]
            }
            self.dynamic_shape.opt_input_shape = {
                "input_data": [1, 64, 64, 3 * attrs[0]['groups']],
                "output_data": [1, 64, 64, 24]
            }

        def clear_dynamic_shape():
            self.dynamic_shape.min_input_shape = {}
            self.dynamic_shape.max_input_shape = {}
            self.dynamic_shape.opt_input_shape = {}

        def generate_trt_nodes_num(attrs, dynamic_shape):
            return 1, 2

        attrs = [
            program_config.ops[i].attrs
            for i in range(len(program_config.ops))
        ]

        # for static_shape
        clear_dynamic_shape()
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), (1e-5, 1e-5)
        self.trt_param.precision = paddle_infer.PrecisionType.Int8
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, False), (1e-5, 1e-5)

        # for dynamic_shape
        generate_dynamic_shape(attrs)
        self.trt_param.precision = paddle_infer.PrecisionType.Float32
        yield self.create_inference_config(), generate_trt_nodes_num(attrs,
                                                                     True), 1e-5
        self.trt_param.precision = paddle_infer.PrecisionType.Half
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), (1e-5, 1e-5)
        self.trt_param.precision = paddle_infer.PrecisionType.Int8
        yield self.create_inference_config(), generate_trt_nodes_num(
            attrs, True), (1e-5, 1e-5)

    def test(self):
        self.run_test()

    # Paddle raw OP fake_channel_wise_dequantize_abs_max dose not support NHWC data format.
    # GPU (baseline) will lead to an error (incorrect channel size).
    # Paddle-TRT will convert IR pattern quant->conv2d->dequant to conv2d (with attr InScale), so it can work.
    # Hence, we only test whether Paddle-TRT can successfully work or not.
    def test_quant(self):
        status = True
        np.random.seed(int(time.strftime("%W")))
        run_flags = []
        for prog_config in self.sample_program_configs():
            # In CI, only run 30% cases
            if np.random.rand() < self.num_percent_cases:
                run_flags.append(True)
            else:
                run_flags.append(False)
        np.random.seed(1024)
        for prog_config, run_flags in zip(self.sample_program_configs(),
                                          run_flags):
            if not run_flags:
                continue

            # if program is invalid, we should skip that cases.
            if not self.is_program_valid(prog_config):
                continue
            model, params = create_fake_model(prog_config)
            model, params = create_quant_model(model, params)
            feed_data = {}
            for name, tensor_config in prog_config.inputs.items():
                feed_data[name] = {
                    'data': tensor_config.data,
                    'lod': tensor_config.lod
                }
            for pred_config, nodes_num, threshold in self.sample_predictor_configs(
                    prog_config):
                if os.path.exists(self.trt_cache_dir):
                    shutil.rmtree(self.trt_cache_dir)

                if pred_config.tensorrt_precision_mode(
                ) != paddle_infer.PrecisionType.Int8:
                    continue

                try:
                    pred_config_deserialize = paddle_infer.Config(pred_config)
                    self.run_test_config(model, params, prog_config,
                                         pred_config, feed_data)
                    self.assert_op_size(nodes_num[0], nodes_num[1])
                    # deserialize test
                    if nodes_num[0] > 0:
                        self.run_test_config(model, params, prog_config,
                                             pred_config_deserialize, feed_data)
                except Exception as e:
                    self.fail_log(
                        str(prog_config) + ' vs ' + self.inference_config_str(
                            pred_config) +
                        '\033[1;31m \nERROR INFO: {}\033[0m'.format(str(e)))
                    status = False
                    continue
                self.success_log('RUN ' + str(prog_config) + ' with ' +
                                 self.inference_config_str(pred_config))

        self.assertTrue(status)


if __name__ == "__main__":
    unittest.main()
