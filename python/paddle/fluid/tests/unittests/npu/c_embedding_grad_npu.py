#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import numpy as np

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle import _C_ops

SEED = 2021


def get_c_lookup_tablev2_grad(dtype):
    device = paddle.set_device('npu:0')
    start = 2
    end = 12
    np.random.seed(SEED)
    ids = np.random.randint(low=0, high=20, size=(6, 8)).astype(np.int32)
    index = ids.flatten()
    input_mask = (index < start) | (index >= end)
    masked_input = index - start
    masked_input[input_mask] = 10
    masked_input.resize(6, 8)
    table = np.zeros([11, 64]).astype(dtype)
    tmp = np.ones([10, 64]).astype(dtype)
    table[:10, :] = tmp[:]
    table = paddle.to_tensor(table, stop_gradient=False)
    masked_input = paddle.to_tensor(masked_input, stop_gradient=False)
    c_embedding = _C_ops.lookup_table_v2(table, masked_input)
    c_embedding.backward()
    return table.grad.numpy()[:-1]


def _get_c_embedding_grad(dtype):
    device = paddle.set_device('npu:0')
    start = 2
    np.random.seed(SEED)
    ids = np.random.randint(low=0, high=20, size=(6, 8)).astype(np.int32)
    table = np.ones([10, 64]).astype(dtype)
    table = paddle.to_tensor(table, stop_gradient=False)
    ids = paddle.to_tensor(ids, stop_gradient=False)
    c_embedding = _C_ops.c_embedding(table, ids, 'start_index', start)
    c_embedding.backward()
    return (table.grad.numpy() == get_c_lookup_tablev2_grad(dtype)).all


def get_c_embedding_grad():
    result = []
    for table_dtype in [np.float32, np.float16]:
        res = _get_c_embedding_grad(table_dtype)
        assert res, "c_embedding_grad {} dtype find error!".format(table_dtype)
        result.append(res)
    return np.array(result).all()
