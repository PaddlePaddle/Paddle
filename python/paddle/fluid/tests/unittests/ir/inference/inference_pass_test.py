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
    def __init__(self, methodName='runTest'):
        super(InferencePassTest, self).__init__(methodName)
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()
        self.feeds = None
        self.fetch_list = None

        self.enable_mkldnn = False
        self.enable_trt = False
        self.trt_parameters = None
        self.enable_lite = False
        self.lite_parameters = None
        self.path = "./inference_pass/" + self.__class__.__name__ + "/"
        np.random.seed(1)
        random.seed(1)

    def _get_place(self):
        return set([False, core.is_compiled_with_cuda()])

    def _save_models(self, executor, program):
        outs = executor.run(program=program,
                            feed=self.feeds,
                            fetch_list=self.fetch_list,
                            return_numpy=False)
        # save models as combined to ensure that 
        # there won't be too many useless files 
        # after finishing a couple of tests.
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
        for i, name in enumerate(names):
            shape = tensor_shapes[name]
            shape[0] = 1
            tensor = predictor.get_input_tensor(name)
            tensor.copy_from_cpu(list(self.feeds.values())[i])

        predictor.zero_copy_run()

        output_names = predictor.get_output_names()
        outs = [
            predictor.get_output_tensor(out_name).copy_to_cpu()
            for out_name in output_names
        ]

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
        self.assertFalse(self.feeds is None,
                         "The inputs of the model is None. ")
        use_gpu = self._get_place()
        for place_ in use_gpu:
            self.check_output_with_option(place_, atol)

    def check_output_with_option(self, use_gpu, atol=1e-5):
        '''
        Check whether calculating on CPU and GPU, enable TensorRT 
        or disable TensorRT, enable MKLDNN or disable MKLDNN 
        are all the same. 
        '''
        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        executor = fluid.Executor(place)
        device = "GPU" if use_gpu else "CPU"
        executor.run(self.startup_program)
        outs = self._save_models(executor, self.main_program)

        analysis_outputs = self._get_analysis_outputs(
            self._get_analysis_config(use_gpu=use_gpu))

        # Check whether the results calculated on CPU and on GPU are the same. 
        self.assertTrue(
            len(outs) == len(analysis_outputs),
            "The number of outputs is different between inference and training forward at {}".
            format(device))

        for out, analysis_output in zip(outs, analysis_outputs):
            self.assertTrue(
                np.allclose(
                    np.array(out), analysis_output, atol=atol),
                "Output has diff between inference and training forward at {} ".
                format(device))

        # Check whether the trt results and the GPU results are the same. 
        if use_gpu and self.enable_trt:
            tensorrt_outputs = self._get_analysis_outputs(
                self._get_analysis_config(
                    use_gpu=use_gpu, use_trt=self.enable_trt))

            self.assertTrue(
                len(tensorrt_outputs) == len(outs),
                "The number of outputs is different between GPU and TensorRT. ")

            for out, tensorrt_output in zip(outs, tensorrt_outputs):
                self.assertTrue(
                    np.allclose(
                        np.array(out), tensorrt_output, atol=atol),
                    "Output has diff between GPU and TensorRT. ")

        # Check whether the mkldnn results and the CPU results are the same. 
        if (not use_gpu) and self.enable_mkldnn:
            mkldnn_outputs = self._get_analysis_outputs(
                self._get_analysis_config(
                    use_gpu=use_gpu, use_mkldnn=self.enable_mkldnn))

            self.assertTrue(
                len(outs) == len(mkldnn_outputs),
                "The number of outputs is different between CPU and MKLDNN. ")

            for out, mkldnn_output in zip(outs, mkldnn_outputs):
                self.assertTrue(
                    np.allclose(
                        np.array(out), mkldnn_output, atol=atol),
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
