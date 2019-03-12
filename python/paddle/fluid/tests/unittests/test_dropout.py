# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle.fluid as fluid
from paddle.fluid.imperative import Embedding, LayerNorm, FC, to_variable, Layer, guard
import paddle.fluid.framework as framework
from paddle.fluid.optimizer import SGDOptimizer
from test_imperative_base import new_program_scope
import numpy as np
import pdb
import six
with guard():
    npd = np.arange(1.0, 4.0).astype("float32")
    enc_input = to_variable(npd)
    b = fluid.layers.dropout(
        enc_input, dropout_prob=0.01, seed=None, is_test=False)
    print(b)
