# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import six
import random
import unittest
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import PaddleDType
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor


class InferencePassTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()
        self.feeds = None
        self.fetch_list = None

        self.enable_mkldnn = False
        self.enable_trt = False
        self.trt_parameters = None
        self.enable_lite = False
        self.lite_parameters = None
        self.path = "./inference_pass/"
        np.random.seed(1)
        random.seed(1)

    def _get_place(self):
        use_gpu = [False]
        if core.is_compiled_with_cuda():
            use_gpu.append(True)
        return use_gpu

    def _save_models(self, executor, program):
        outs = executor.run(program=program,
                            feed=self.feeds,
                            fetch_list=self.fetch_list,
                            return_numpy=False)
        fluid.io.save_inference_model(
            dirname=self.path,
            feeded_var_names=list(self.feeds.keys()),
            target_vars=self.fetch_list,
            executor=executor,
            main_program=program,
            model_filename="model",
            params_filename="params")

        return outs

    def _get_analysis_outputs(self, config):
        '''
        Return AnalysisPredictor outputs. 
        '''
        predictor = create_paddle_predictor(config)
        tensor_shapes = predictor.get_input_tensor_shape()
        names = predictor.get_input_names()
        for i in range(len(names)):
            shape = tensor_shapes[names[i]]
            shape[0] = 1
            tensor = predictor.get_input_tensor(names[i])
            tensor.copy_from_cpu(list(self.feeds.values())[i])

        predictor.zero_copy_run()

        output_names = predictor.get_output_names()
        outs = []
        for out_name in output_names:
            output_tensor = predictor.get_output_tensor(out_name)
            outs.append(output_tensor.copy_to_cpu())

        return outs

    def _get_analysis_config(self,
                             use_gpu=False,
                             use_trt=False,
                             use_mkldnn=False):
        '''
        Return a new object of AnalysisConfig. 
        '''
        config = AnalysisConfig(
            os.path.join(self.path, "model"), os.path.join(self.path, "params"))
        config.disable_gpu()
        config.switch_specify_input_names(True)
        config.switch_ir_optim(True)
        config.switch_use_feed_fetch_ops(False)
        if use_gpu:
            config.enable_use_gpu(100, 0)
            if use_trt:
                config.enable_tensorrt_engine(
                    self.trt_parameters.workspace_size,
                    self.trt_parameters.max_batch_size,
                    self.trt_parameters.min_subgraph_size,
                    self.trt_parameters.precision,
                    self.trt_parameters.use_static,
                    self.trt_parameters.use_calib_mode)
        elif use_mkldnn:
            config.enable_mkldnn()

        return config

    def check_output(self, atol=1e-5):
        '''
        Check whether calculating on CPU and GPU, enable TensorRT 
        or disable TensorRT, enable MKLDNN or disable MKLDNN 
        are all the same. 
        '''
        if self.feeds is None:
            self.assertTrue(False, "The inputs of the model is None. ")
        use_gpu = self._get_place()
        for i in range(len(use_gpu)):
            self.check_output_with_option(use_gpu[i], atol)

    def check_output_with_option(self, use_gpu, atol=1e-5):
        '''
        Check whether calculating on CPU and GPU, enable TensorRT 
        or disable TensorRT, enable MKLDNN or disable MKLDNN 
        are all the same. 
        '''
        if use_gpu:
            executor = fluid.Executor(fluid.CUDAPlace(0))
            place = "GPU"
        else:
            executor = fluid.Executor(fluid.CPUPlace())
            place = "CPU"
        executor.run(self.startup_program)
        outs = self._save_models(executor, self.main_program)

        analysis_outputs = self._get_analysis_outputs(
            self._get_analysis_config(use_gpu=use_gpu))

        # Check whether the results calculated on CPU and on GPU are the same. 
        self.assertTrue(
            len(outs) == len(analysis_outputs),
            "The number of outputs is different between inference and training forward at {}".
            format(place))

        for i in six.moves.xrange(len(outs)):
            self.assertTrue(
                np.allclose(
                    np.array(outs[i]), analysis_outputs[i], atol=atol),
                "Output has diff between inference and training forward at {} ".
                format(place))

        # Check whether the trt results and the GPU results are the same. 
        if use_gpu and self.enable_trt:
            tensorrt_outputs = self._get_analysis_outputs(
                self._get_analysis_config(
                    use_gpu=use_gpu, use_trt=self.enable_trt))

            self.assertTrue(
                len(tensorrt_outputs) == len(outs),
                "The number of outputs is different between GPU and TensorRT. ")

            for i in six.moves.xrange(len(outs)):
                self.assertTrue(
                    np.allclose(
                        tensorrt_outputs[i], np.array(outs[i]), atol=atol),
                    "Output has diff between GPU and TensorRT. ")

        # Check whether the mkldnn results and the CPU results are the same. 
        if (not use_gpu) and self.enable_mkldnn:
            mkldnn_outputs = self._get_analysis_outputs(
                self._get_analysis_config(
                    use_gpu=use_gpu, use_mkldnn=self.enable_mkldnn))

            self.assertTrue(
                len(outs) == len(mkldnn_outputs),
                "The number of outputs is different between CPU and MKLDNN. ")

            for i in six.moves.xrange(len(outs)):
                self.assertTrue(
                    np.allclose(
                        np.array(outs[i]), mkldnn_outputs[i], atol=atol),
                    "Output has diff between CPU and MKLDNN. ")

    class TensorRTParam:
        '''
        Prepare TensorRT subgraph engine parameters. 
        '''

        def __init__(self, workspace_size, max_batch_size, min_subgraph_size,
                     precision, use_static, use_calib_mode):
            self.workspace_size = workspace_size
            self.max_batch_size = max_batch_size
            self.min_subgraph_size = min_subgraph_size
            self.precision = precision
            self.use_static = use_static
            self.use_calib_mode = use_calib_mode

    class LiteParam:
        '''
        Prepare Lite subgraph engine parameters. 
        '''

        def __init__(self, precision, passes_filter, ops_filter):
            self.precision = precision
            self.passes_filter = passes_filter
            self.ops_filter = ops_filter
