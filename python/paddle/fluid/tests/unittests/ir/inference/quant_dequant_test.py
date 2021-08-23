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

import os
import unittest
import random
import numpy as np
import six
import paddle.fluid as fluid
import paddle
import warnings
from paddle.fluid.framework import IrGraph
from paddle.fluid.contrib.slim.quantization import QuantizationTransformPass
from paddle.fluid.contrib.slim.quantization import QuantizationFreezePass
from paddle.fluid.contrib.slim.quantization import OutScaleForTrainingPass
from paddle.fluid.contrib.slim.quantization import OutScaleForInferencePass
from paddle.fluid.contrib.slim.quantization import AddQuantDequantPass
from paddle.fluid import (core, Program, Variable, program_guard, layers)
from paddle.fluid.io import prepend_feed_ops, append_fetch_ops
from inference_pass_test import InferencePassTest
from paddle.fluid.core import create_paddle_predictor
from paddle.fluid.core import AnalysisConfig


class QuantDequantTest(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super(QuantDequantTest, self).__init__(methodName)
        paddle.enable_static()
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()
        self.test_main_program = fluid.Program()
        self.test_startup_program = fluid.Program()
        self.feeds = None
        self.fetch_list = None
        self.enable_mkldnn = False
        self.enable_mkldnn_bfloat16 = False
        self.enable_trt = False
        self.enable_tensorrt_oss = True
        self.trt_parameters = None
        self.dynamic_shape_params = None
        self.enable_lite = False
        self.lite_parameters = None
        self.path = "./inference_pass/" + self.__class__.__name__ + "/"
        self.data = None
        self.label = None
        self.result = None
        np.random.seed(1)
        random.seed(1)

    # from Paddle release2.1 
    def _normalize_program(self, program, feed_vars, fetch_vars):
        if not isinstance(program, Program):
            raise TypeError(
                "program type must be `fluid.Program`, but received `%s`" %
                type(program))
        if not isinstance(feed_vars, list):
            feed_vars = [feed_vars]
        if not all(isinstance(v, Variable) for v in feed_vars):
            raise TypeError(
                "feed_vars type must be a Variable or a list of Variable.")
        if not isinstance(fetch_vars, list):
            fetch_vars = [fetch_vars]
        if not all(isinstance(v, Variable) for v in fetch_vars):
            raise TypeError(
                "fetch_vars type must be a Variable or a list of Variable.")

        # remind users to set auc_states to 0 if auc op were found.
        for op in program.global_block().ops:
            # clear device of Op
            device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName(
            )
            op._set_attr(device_attr_name, "")
            if op.type == 'auc':
                warnings.warn("Be sure that you have set auc states to 0 "
                              "before saving inference model.")
                break

        # serialize program
        copy_program = program.clone()
        global_block = copy_program.global_block()
        remove_op_idx = []
        for i, op in enumerate(global_block.ops):
            op.desc.set_is_target(False)
            if op.type == "feed" or op.type == "fetch":
                remove_op_idx.append(i)
        for idx in remove_op_idx[::-1]:
            global_block._remove_op(idx)
        copy_program.desc.flush()

        feed_var_names = [var.name for var in feed_vars]
        copy_program = copy_program._prune_with_input(
            feeded_var_names=feed_var_names, targets=fetch_vars)
        copy_program = copy_program._inference_optimize(prune_read_op=True)
        fetch_var_names = [var.name for var in fetch_vars]
        prepend_feed_ops(copy_program, feed_var_names)
        append_fetch_ops(copy_program, fetch_var_names)
        copy_program.desc._set_version()
        return copy_program

    def _save_models(self, dirname, feeded_var_names, target_vars, executor,
                     program, scope):
        with fluid.scope_guard(scope):
            fluid.io.save_inference_model(dirname, feeded_var_names,
                                          target_vars, executor, program)

    def _get_paddle_outs(self, feed, fetch_list, executor, program, scope):
        '''
        Return PaddlePaddle outputs. 
        '''
        with fluid.scope_guard(scope):
            outs = executor.run(program=program,
                                feed=feed,
                                fetch_list=fetch_list,
                                return_numpy=True)
        return outs

    def _get_inference_outs(self, config):
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
            feed_data = list(self.feeds.values())[i]
            tensor.copy_from_cpu(np.array(feed_data))
            if type(feed_data) == fluid.LoDTensor:
                tensor.set_lod(feed_data.lod())

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
        config = AnalysisConfig(self.path)
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

                if self.dynamic_shape_params:
                    config.set_trt_dynamic_shape_info(
                        self.dynamic_shape_params.min_input_shape,
                        self.dynamic_shape_params.max_input_shape,
                        self.dynamic_shape_params.optim_input_shape,
                        self.dynamic_shape_params.disable_trt_plugin_fp16)
                if self.enable_tensorrt_oss:
                    config.enable_tensorrt_oss()

        elif use_mkldnn:
            config.enable_mkldnn()
            if self.enable_mkldnn_bfloat16:
                config.enable_mkldnn_bfloat16()
        print('config summary:', config.summary())
        return config

    def check_output_with_option(self,
                                 use_gpu,
                                 atol=1e-5,
                                 flatten=False,
                                 quant=False,
                                 rtol=1e-5):
        '''
        Check whether calculating on CPU and GPU, enable TensorRT 
        or disable TensorRT, enable MKLDNN or disable MKLDNN 
        are all the same. 
        '''
        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        executor = fluid.Executor(place)
        scope = fluid.Scope()
        device = "GPU" if use_gpu else "CPU"

        with fluid.scope_guard(scope):
            executor.run(self.startup_program)
            executor.run(self.test_startup_program)
        main_graph = IrGraph(core.Graph(self.main_program.desc), for_test=False)
        test_graph = IrGraph(
            core.Graph(self.test_main_program.desc), for_test=True)

        transform_pass = QuantizationTransformPass(
            scope=scope,
            place=place,
            activation_quantize_type=self.activation_quantize_type,
            weight_quantize_type=self.weight_quantize_type)
        transform_pass.apply(main_graph)
        transform_pass.apply(test_graph)

        add_quant_dequant_pass = AddQuantDequantPass(scope=scope, place=place)
        add_quant_dequant_pass.apply(main_graph)
        add_quant_dequant_pass.apply(test_graph)

        scale_training_pass = OutScaleForTrainingPass(scope=scope, place=place)
        scale_training_pass.apply(main_graph)

        build_strategy = fluid.BuildStrategy()
        build_strategy.memory_optimize = False
        build_strategy.enable_inplace = False
        build_strategy.fuse_all_reduce_ops = False
        binary = fluid.CompiledProgram(main_graph.graph)

        iters = 10
        batch_size = 1
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.mnist.train(), buf_size=500),
            batch_size=batch_size)
        feeder = fluid.DataFeeder(
            feed_list=[self.data, self.label], place=place)
        with fluid.scope_guard(scope):
            for _ in range(iters):
                data = next(train_reader())
                loss_v = executor.run(binary,
                                      feed=feeder.feed(data),
                                      fetch_list=[self.loss])

        scale_inference_pass = OutScaleForInferencePass(scope=scope)
        scale_inference_pass.apply(test_graph)

        # Freeze graph for inference, but the weight of fc/conv is still float type.
        freeze_pass = QuantizationFreezePass(
            scope=scope,
            place=place,
            weight_quantize_type=self.weight_quantize_type)
        freeze_pass.apply(test_graph)

        self.main_program = test_graph.to_program()

        with fluid.scope_guard(scope):
            self.main_program = self._normalize_program(
                self.main_program, self.data, self.fetch_list)

        self._save_models(self.path,
                          list(self.feeds.keys()), self.fetch_list, executor,
                          self.main_program, scope)

        paddle_outs = self._get_paddle_outs(self.feeds, self.fetch_list,
                                            executor, self.main_program, scope)
        inference_outs = self._get_inference_outs(
            self._get_analysis_config(use_gpu=use_gpu))

        # Check whether the results calculated on CPU and on GPU are the same. 
        self.assertTrue(
            len(paddle_outs) == len(inference_outs),
            "The number of outputs is different between inference and training forward at {}".
            format(device))

        for out, inference_out in zip(paddle_outs, inference_outs):
            paddle_out = np.array(out)

            if flatten:
                paddle_out = paddle_out.flatten()
                inference_out = inference_out.flatten()

            self.assertTrue(
                np.allclose(
                    paddle_out, inference_out, atol=atol),
                "Output has diff between inference and training forward at {} ".
                format(device))

        # Check whether the trt results and the GPU results are the same. 
        if use_gpu and self.enable_trt:
            tensorrt_outputs = self._get_inference_outs(
                self._get_analysis_config(
                    use_gpu=use_gpu, use_trt=self.enable_trt))

            if self.trt_parameters.use_static:
                #deserialize
                tensorrt_outputs = self._get_inference_outs(
                    self._get_analysis_config(
                        use_gpu=use_gpu, use_trt=self.enable_trt))

            self.assertTrue(
                len(tensorrt_outputs) == len(paddle_outs),
                "The number of outputs is different between GPU and TensorRT. ")

            for paddle_out, tensorrt_output in zip(paddle_outs,
                                                   tensorrt_outputs):
                paddle_out = np.array(paddle_out)

                if flatten:
                    paddle_out = paddle_out.flatten()
                    tensorrt_output = tensorrt_output.flatten()

                self.assertTrue(
                    np.allclose(
                        paddle_out, tensorrt_output, rtol=rtol, atol=atol),
                    "Output has diff between GPU and TensorRT. ")

        # Check whether the mkldnn results and the CPU results are the same. 
        if (not use_gpu) and self.enable_mkldnn:
            mkldnn_outputs = self._get_inference_outs(
                self._get_analysis_config(
                    use_gpu=use_gpu, use_mkldnn=self.enable_mkldnn))

            self.assertTrue(
                len(paddle_outs) == len(mkldnn_outputs),
                "The number of outputs is different between CPU and MKLDNN. ")

            if self.enable_mkldnn_bfloat16:
                atol = 0.01
            for paddle_out, mkldnn_output in zip(paddle_outs, mkldnn_outputs):
                self.assertTrue(
                    np.allclose(
                        np.array(paddle_out), mkldnn_output, atol=atol),
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

    def quant_dequant(self):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        scope = fluid.Scope()
