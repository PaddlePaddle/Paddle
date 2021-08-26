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
import abc
import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import NumpyArrayInitializer
import paddle.fluid.core as core
from paddle import compat as cpt
import paddle.inference as paddle_infer
from typing import Optional, List, Callable, Dict, Any, Set
from program_config import TensorConfig, OpConfig, ProgramConfig, create_fake_model, create_quant_model


class AutoScanTest(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        paddle.enable_static()
        super(AutoScanTest, self).__init__(methodName)

    @abc.abstractmethod
    def sample_program_configs(self) -> List[ProgramConfig]:
        '''
        Generate all config with the combination of different Input tensor shape and
        different Attr values.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def sample_predictor_configs(self) -> List[paddle_infer.Config]:
        raise NotImplementedError

    def run_test_config(self, model, params, prog_config, pred_config,
                        feed_data) -> Dict[str, np.ndarray]:
        '''
        Test a single case.
        '''
        pred_config.set_model_buffer(model, len(model), params, len(params))
        predictor = paddle_infer.create_predictor(pred_config)

        for name, _ in prog_config.inputs.items():
            input_tensor = predictor.get_input_handle(name)
            input_tensor.copy_from_cpu(feed_data[name])
        predictor.run()
        result = {}
        for out_name, o_name in zip(prog_config.outputs,
                                    predictor.get_output_names()):
            result[out_name] = predictor.get_output_handle(o_name).copy_to_cpu()
        return result

    def assert_op_size(self, trt_engine_num, paddle_op_num):
        cur_path = os.path.dirname(__file__)
        last_passed_program = os.path.join(
            cur_path, 'transpose_flatten_concat_fuse_pass.pdmodel')
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

    def assert_tensors_near(self,
                            threshold: float,
                            tensors: List[Dict[str, np.array]]):
        assert len(tensors) > 1
        first = tensors[0]
        for group in tensors[1:]:
            for key, arr in group.items():
                self.assertTrue(
                    np.allclose(
                        first[key], arr, atol=threshold),
                    "Output has diff between GPU and TensorRT. ")

    def run_test(self,
                 trt_engine_num: int,
                 paddle_op_num: int,
                 threshold=1e-5,
                 quant=False):
        for prog_config in self.sample_program_configs():
            model, params = create_fake_model(prog_config)
            if quant:
                model, params = create_quant_model(model, params)
            for batch_size in self.batch_size_set:
                feed_data = {}
                for name, tensor_config in prog_config.inputs.items():
                    tensor_shape = tensor_config.shape.copy()
                    tensor_shape[0] = batch_size
                    feed_data[name] = np.random.random(tensor_shape).astype(
                        tensor_config.dtype)
                results: List[Dict[str, Tensor]] = []
                for pred_config in self.sample_predictor_configs():
                    results.append(
                        self.run_test_config(model, params, prog_config,
                                             pred_config, feed_data))
                self.assert_tensors_near(threshold=threshold, tensors=results)
                self.assert_op_size(trt_engine_num, paddle_op_num)
