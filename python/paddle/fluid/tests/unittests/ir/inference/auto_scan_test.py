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
import enum
import logging
import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import NumpyArrayInitializer
import paddle.fluid.core as core
from paddle import compat as cpt
import paddle.inference as paddle_infer
from typing import Optional, List, Callable, Dict, Any, Set
from program_config import TensorConfig, OpConfig, ProgramConfig, create_fake_model, create_quant_model

logging.basicConfig(level=logging.INFO, format="%(message)s")


class SkipReasons(enum.Enum):
    # Paddle not support, but trt support, we need to add the feature.
    TRT_NOT_IMPLEMENTED = 0
    # TRT not support.
    TRT_NOT_SUPPORT = 1


class AutoScanTest(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        np.random.seed(1024)
        paddle.enable_static()
        super(AutoScanTest, self).__init__(methodName)
        self.skip_cases = []

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

    @abc.abstractmethod
    def add_skip_case(
            self,
            teller: [Callable[[ProgramConfig, paddle_infer.Config], bool]],
            reason: SkipReasons,
            note: str):
        self.skip_cases.append((teller, reason, note))

    @abc.abstractmethod
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
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
            input_tensor.copy_from_cpu(feed_data[name]['data'])
            if feed_data[name]['lod'] is not None:
                input_tensor.set_lod(feed_data[name]['lod'])
        predictor.run()
        result = {}
        for out_name, o_name in zip(prog_config.outputs,
                                    predictor.get_output_names()):
            result[out_name] = predictor.get_output_handle(o_name).copy_to_cpu()
        return result

    def assert_tensors_near(self,
                            threshold: float,
                            tensors: List[Dict[str, np.array]]):
        assert len(tensors) > 1
        first = tensors[0]
        for group in tensors[1:]:
            for key, arr in group.items():
                self.assertTrue(
                    first[key].shape == arr.shape,
                    "The output shape of GPU and TensorRT are not equal.")
                self.assertTrue(
                    np.allclose(
                        first[key], arr, atol=threshold),
                    "Output has diff between GPU and TensorRT. ")

    @abc.abstractmethod
    def run_test(self, quant=False):
        raise NotImplementedError
