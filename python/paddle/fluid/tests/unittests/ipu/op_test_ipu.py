#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import random
import unittest
import numpy as np
from enum import Enum

import paddle
import paddle.static

map_np_dtype_to_fluid_dtype = {
    'bool': "bool",
    'int8': "int8",
    'uint8': "uint8",
    "int32": "int32",
    "int64": "int64",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
}


class ExecutionMode(Enum):
    CPU_FP32 = 1
    IPU_FP32 = 2
    # enable_fp16 through ipu_strategy.enable_fp16
    IPU_POPART_FP16 = 3

    def __lt__(self, other):
        return self.value < other.value

    def __gt__(self, other):
        return self.value > other.value


def np_dtype_to_fluid_str(dtype: np.dtype) -> str:
    return map_np_dtype_to_fluid_dtype[dtype.name]


class IPUOpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get random seeds
        cls._np_rand_state = np.random.get_state()
        cls._py_rand_state = random.getstate()

        cls.SEED = 2021
        np.random.seed(cls.SEED)
        random.seed(cls.SEED)

        # Enable paddle static graph mode
        paddle.enable_static()

    @classmethod
    def tearDownClass(cls):
        """Restore random seeds"""
        np.random.set_state(cls._np_rand_state)
        random.setstate(cls._py_rand_state)

    @classmethod
    def use_ipumodel(cls):
        if 'POPLAR_IPUMODEL' not in os.environ:
            return False
        else:
            flag = os.environ['POPLAR_IPUMODEL']
            if flag.upper() in ['1', "TRUE"]:
                return True

    def set_atol(self):
        self.atol = 1e-10
        self.rtol = 1e-6
        self.atol_fp16 = 1e-3
        self.rtol_fp16 = 1e-3

    def set_training(self):
        self.is_training = False
        self.epoch = 1

    def check(self, outputs, check_shape=False):
        cpu_fp32 = outputs[ExecutionMode.CPU_FP32]
        ipu_fp32 = outputs[ExecutionMode.IPU_FP32]
        max_diff = np.abs(cpu_fp32 - ipu_fp32).max()
        fp32_flag = np.allclose(
            cpu_fp32, ipu_fp32, rtol=self.rtol, atol=self.atol)
        self.assertTrue(fp32_flag, "max diff is %f" % (max_diff))

        if check_shape:
            self.assertTrue(cpu_fp32.shape == ipu_fp32.shape)

        ipu_popart_fp16 = None
        if ExecutionMode.IPU_POPART_FP16 in outputs.keys():
            ipu_popart_fp16 = outputs[ExecutionMode.IPU_POPART_FP16]
            max_diff = np.abs(ipu_popart_fp16.astype(np.float32) -
                              cpu_fp32).max()
            fp16_flag = np.allclose(
                ipu_popart_fp16.astype(np.float32),
                cpu_fp32,
                rtol=self.rtol_fp16,
                atol=self.atol_fp16)
            self.assertTrue(fp16_flag, "max diff is %f" % (max_diff))

            if check_shape:
                self.assertTrue(ipu_popart_fp16.shape == cpu_fp32.shape)
