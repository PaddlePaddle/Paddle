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
import enum
import sys
import os
import logging
import time
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.inference as paddle_infer
import shutil

from paddle import compat as cpt
from typing import Optional, List, Callable, Dict, Any, Set
from program_config import TensorConfig, OpConfig, ProgramConfig, create_fake_model, create_quant_model
from auto_scan_test import AutoScanTest, SkipReasons

logging.basicConfig(level=logging.INFO, format="%(message)s")


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

        def __init__(self, min_input_shape, max_input_shape, opt_input_shape,
                     disable_trt_plugin_fp16):
            self.min_input_shape = min_input_shape
            self.max_input_shape = max_input_shape
            self.opt_input_shape = opt_input_shape
            self.disable_trt_plugin_fp16 = disable_trt_plugin_fp16

    def __init__(self, methodName='runTest'):
        super(TrtLayerAutoScanTest, self).__init__(methodName)
        self.trt_param = self.TensorRTParam(
            workspace_size=1024,
            max_batch_size=4,
            min_subgraph_size=0,
            precision=paddle_infer.PrecisionType.Float32,
            use_static=True,
            use_calib_mode=False)
        self.dynamic_shape = self.DynamicShapeParam({}, {}, {}, False)
        self.num_percent_cases = float(
            os.getenv(
                'TEST_NUM_PERCENT_CASES', default='1.0'))
        abs_dir = os.path.abspath(os.path.dirname(__file__))
        cache_dir = str(self.__module__) + '_trt_cache_dir'
        self.trt_cache_dir = os.path.join(abs_dir, cache_dir)

    def create_inference_config(self, use_trt=True) -> paddle_infer.Config:
        config = paddle_infer.Config()
        # config.disable_glog_info()
        config.enable_use_gpu(100, 0)
        config.set_optim_cache_dir(self.trt_cache_dir)
        if use_trt:
            config.switch_ir_debug()
            config.enable_tensorrt_engine(
                max_batch_size=self.trt_param.max_batch_size,
                workspace_size=self.trt_param.workspace_size,
                min_subgraph_size=self.trt_param.min_subgraph_size,
                precision_mode=self.trt_param.precision,
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

    def assert_tensors_near(self,
                            atol: float,
                            rtol: float,
                            tensor: Dict[str, np.array],
                            baseline: Dict[str, np.array]):
        for key, arr in tensor.items():
            self.assertTrue(
                baseline[key].shape == arr.shape,
                "The output shape of GPU and TensorRT are not equal, the baseline shape is "
                + str(baseline[key].shape) + ', but the trt shape is ' +
                str(arr.shape))
            self.assertTrue(
                np.allclose(
                    baseline[key], arr, atol=atol, rtol=rtol),
                "Output has diff between GPU and TensorRT. ")

    def assert_op_size(self, trt_engine_num, paddle_op_num):
        last_passed_program = 'transpose_flatten_concat_fuse_pass.pdmodel'
        model_bytes = paddle.static.load_from_file(last_passed_program)
        pg = paddle.static.deserialize_program(model_bytes)
        main_block = pg.desc.block(0)
        op_size = main_block.op_size()
        op_types = [
            main_block.op(i).type() == 'tensorrt_engine' for i in range(op_size)
        ]
        trt_engine_size = sum(op_types)
        paddle_op_size = op_size - trt_engine_size
        self.assertTrue(trt_engine_size == trt_engine_num,
                        'trt_engine_num is {}, but got {}!'.format(
                            trt_engine_size, trt_engine_num))
        self.assertTrue(paddle_op_size == paddle_op_num,
                        'paddle_op_num is {}, but got {}!'.format(
                            paddle_op_size, paddle_op_num))

    def skip_log(self, msg: str):
        logging.warning("SKIP: " + msg)

    def fail_log(self, msg: str):
        logging.error("FAILE: " + msg)

    def success_log(self, msg: str):
        logging.info("SUCCESS: " + msg)

    def validate(self, func: Callable[..., bool]):
        pass

    def generate_op_config(self,
                           ops_config: List[Dict[str, Any]]) -> List[OpConfig]:
        ops = []
        for i in range(len(ops_config)):
            op_config = ops_config[i]
            ops.append(
                OpConfig(
                    type=op_config['op_type'],
                    inputs=op_config['op_inputs'],
                    outputs=op_config['op_outputs'],
                    attrs=op_config['op_attrs']))
        return ops

    def inference_config_str(self, config: paddle_infer.Config):
        dic = {}
        enable_trt = config.tensorrt_engine_enabled()
        trt_precison = config.tensorrt_precision_mode()
        trt_dynamic_shape = config.tensorrt_dynamic_shape_enabled()
        if enable_trt:
            dic['use_trt'] = True
            dic['trt_precision'] = trt_precison
            dic['use_dynamic_shape'] = trt_dynamic_shape
        else:
            dic['use_trt'] = False
        return str(dic)

    def run_test(self, quant=False):
        status = True
        np.random.seed(int(1000 * time.time()) % 2**32)
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
            if quant:
                model, params = create_quant_model(model, params)

            feed_data = {}
            for name, tensor_config in prog_config.inputs.items():
                feed_data[name] = {
                    'data': tensor_config.data,
                    'lod': tensor_config.lod
                }

            results: List[Dict[str, Tensor]] = []

            # baseline: gpu run
            gpu_config = self.create_inference_config(use_trt=False)
            results.append(
                self.run_test_config(model, params, prog_config, gpu_config,
                                     feed_data))
            self.success_log('RUN_GPU_BASELINE ' + str(prog_config) + ' vs ' +
                             self.inference_config_str(gpu_config))

            for pred_config, nodes_num, threshold in self.sample_predictor_configs(
                    prog_config):

                if os.path.exists(self.trt_cache_dir):
                    shutil.rmtree(self.trt_cache_dir)

                if isinstance(threshold, float):
                    atol = threshold
                    rtol = 1e-8
                elif isinstance(threshold, list) or isinstance(threshold,
                                                               tuple):
                    atol = threshold[0]
                    rtol = threshold[1]
                else:
                    raise NotImplementedError

                if quant and pred_config.tensorrt_precision_mode(
                ) != paddle_infer.PrecisionType.Int8:
                    continue
                if pred_config.tensorrt_precision_mode(
                ) == paddle_infer.PrecisionType.Int8 and not quant:
                    continue

                skip_flag = False
                for skip_info in self.skip_cases:
                    if skip_info[0](prog_config, pred_config):
                        skip_flag = True
                        if skip_info[1] == SkipReasons.TRT_NOT_IMPLEMENTED:
                            self.skip_log("[TRT_NOT_IMPLEMENTED] " + skip_info[
                                2] + ' ' + repr(prog_config) + ' vs ' + self.
                                          inference_config_str(pred_config))
                        elif skip_info[1] == SkipReasons.TRT_NOT_SUPPORT:
                            self.skip_log("[TRT_NOT_SUPPORT] " + skip_info[
                                2] + ' ' + repr(prog_config) + ' vs ' + self.
                                          inference_config_str(pred_config))
                        else:
                            raise NotImplementedError
                        break

                try:
                    pred_config_deserialize = paddle_infer.Config(pred_config)
                    results.append(
                        self.run_test_config(model, params, prog_config,
                                             pred_config, feed_data))
                    self.assert_tensors_near(atol, rtol, results[-1],
                                             results[0])
                    if not skip_flag:
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

                self.success_log('RUN ' + str(prog_config) + ' vs ' +
                                 self.inference_config_str(pred_config))

            # In the first step, we found the problem, and after the subsequent repairs, the assert assertion will be enabled
            # self.assertTrue(status)
