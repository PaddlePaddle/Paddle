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

import random
import unittest

import numpy as np
from paddle.fluid.tests.unittests.op_test import _set_use_system_allocator
from typing import Optional
import paddle.fluid.compiler as compiler

SEED = 2021

ipu_compiler_ref: Optional[compiler.IPUCompiledProgram] = None

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


def np_dtype_to_fluid_str(dtype: np.dtype) -> str:
    return map_np_dtype_to_fluid_dtype[dtype.name]


class IPUOpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._np_rand_state = np.random.get_state()
        cls._py_rand_state = random.getstate()

        cls.SEED = SEED
        np.random.seed(cls.SEED)
        random.seed(cls.SEED)

        cls._use_system_allocator = _set_use_system_allocator(True)

    @classmethod
    def tearDownClass(cls):
        """Restore random seeds"""
        np.random.set_state(cls._np_rand_state)
        random.setstate(cls._py_rand_state)

        _set_use_system_allocator(cls._use_system_allocator)
        # unittest will to trigger IPUCompiledProgram.__del__ automatically
        global ipu_compiler_ref
        ipu_compiler_ref is not None and ipu_compiler_ref.clean()

    def set_atol(self):
        self.atol = 1e-5

    def set_training(self):
        self.is_training = False
        self.epoch = 1
