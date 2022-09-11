# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import paddle
from paddle import _C_ops
from paddle.fluid import core
from paddle.fluid.framework import _test_eager_guard
import random
import spconv.pytorch as spconv
import torch
import logging
import paddle.incubate as pi
import time
paddle.set_default_dtype("float16")

def generate_data(config):
    values = []
    indices = []
    #nnz = int(config['batch_size'] * config['x'] * config['y'] * config['z'] * (1-config['sparsity']))
    print(config['nnz'])

    for i in range(config['nnz']):
        value = []
        idx = []
        for j in range(config['in_channels']):
            value.append(random.uniform(-1, -0.0001)*random.choice([-1, 1]))
        values.append(value)

        idx.append(random.randrange(0, config['batch_size']))
        idx.append(random.randrange(0, config['x']))
        idx.append(random.randrange(0, config['y']))
        idx.append(random.randrange(0, config['z']))
        indices.append(idx)
    #indices=[[0,1,2,3],[0,1,2,4]]
    #values=[[8,9,10,11],[12,13,14,15]]
    #indices=[[0,1,2,3],[0,1,2,4]]
    #values=[[8],[12]]
    return values, indices


class TestSparseConv(unittest.TestCase):
    def test_conv3d(self):
        paddle.seed(0)
        with _test_eager_guard():
            config = [
                # 0
                {'batch_size': 1,
                 'x': 41,
                 'y': 1600,
                 'z': 1408,
                 'in_channels': 4,
                 'out_channels': 16,
                 'kernel_size': (3, 3, 3),
                 'strides': (1, 1, 1),
                 'paddings': (0, 0, 0),
                 'diff': 1e-3,
                 'nnz': 136000},
                 # 1
                {'batch_size': 1,
                 'x': 41,
                 'y': 1600,
                 'z': 1408,
                 'in_channels': 16,
                 'out_channels': 16,
                 'kernel_size': (3, 3, 3),
                 'strides': (1, 1, 1),
                 'paddings': (0, 0, 0),
                 'diff': 1e-3,
                 'nnz': 136000},
                 # 2
                {'batch_size': 1,
                 'x': 41,
                 'y': 1600,
                 'z': 1408,
                 'in_channels': 16,
                 'out_channels': 32,
                 'kernel_size': (3, 3, 3),
                 'strides': (2, 2, 2),
                 'paddings': (1, 1, 1),
                 'diff': 1e-3,
                 'nnz': 136000},
                 # 3
                {'batch_size': 1,
                 'x': 21,
                 'y': 800,
                 'z': 704,
                 'in_channels': 32,
                 'out_channels': 32,
                 'kernel_size': (3, 3, 3),
                 'strides': (1, 1, 1),
                 'paddings': (0, 0, 0),
                 'diff': 1e-3,
                 'nnz': 220939},
                # 4
                {'batch_size': 1,
                 'x': 21,
                 'y': 800,
                 'z': 704,
                 'in_channels': 32,
                 'out_channels': 32,
                 'kernel_size': (3, 3, 3),
                 'strides': (1, 1, 1),
                 'paddings': (0, 0, 0),
                 'diff': 1e-3,
                 'nnz': 220939},
                 # 5
                {'batch_size': 1,
                 'x': 21,
                 'y': 800,
                 'z': 704,
                 'in_channels': 32,
                 'out_channels': 64,
                 'kernel_size': (3, 3, 3),
                 'strides': (2, 2, 2),
                 'paddings': (1, 1, 1),
                 'diff': 1e-3,
                 'nnz': 220939},
                 # 6
                {'batch_size': 1,
                 'x': 11,
                 'y': 400,
                 'z': 352,
                 'in_channels': 64,
                 'out_channels': 64,
                 'kernel_size': (3, 3, 3),
                 'strides': (1, 1, 1),
                 'paddings': (0, 0, 0),
                 'diff': 1e-2,
                 'nnz': 146376},
                 # 7
                {'batch_size': 1,
                 'x': 11,
                 'y': 400,
                 'z': 352,
                 'in_channels': 64,
                 'out_channels': 64,
                 'kernel_size': (3, 3, 3),
                 'strides': (1, 1, 1),
                 'paddings': (0, 0, 0),
                 'diff': 1e-3,
                 'nnz': 146376},
                 # 8
                {'batch_size': 1,
                 'x': 11,
                 'y': 400,
                 'z': 352,
                 'in_channels': 64,
                 'out_channels': 64,
                 'kernel_size': (3, 3, 3),
                 'strides': (1, 1, 1),
                 'paddings': (0, 0, 0),
                 'diff': 1e-3,
                 'nnz': 146376},
                 # 9
                {'batch_size': 1,
                 'x': 11,
                 'y': 400,
                 'z': 352,
                 'in_channels': 64,
                 'out_channels': 64,
                 'kernel_size': (3, 3, 3),
                 'strides': (1, 1, 1),
                 'paddings': (0, 0, 0),
                 'diff': 1e-3,
                 'nnz': 146376},
                # 10
                {'batch_size': 1,
                 'x': 5,
                 'y': 200,
                 'z': 176,
                 'in_channels': 64,
                 'out_channels': 64,
                 'kernel_size': (3, 3, 3),
                 'strides': (2, 2, 2),
                 'paddings': (0, 1, 1),
                 'diff': 1e-2,
                 'nnz': 65421},
                # 11
                {'batch_size': 1,
                 'x': 5,
                 'y': 200,
                 'z': 176,
                 'in_channels': 64,
                 'out_channels': 64,
                 'kernel_size': (3, 3, 3),
                 'strides': (1, 1, 1),
                 'paddings': (0, 0, 0),
                 'diff': 1e-2,
                 'nnz': 65421},
                 #12
                {'batch_size': 1,
                 'x': 5,
                 'y': 200,
                 'z': 176,
                 'in_channels': 64,
                 'out_channels': 64,
                 'kernel_size': (3, 3, 3),
                 'strides': (1, 1, 1),
                 'paddings': (0, 0, 0),
                 'diff': 1e-3,
                 'nnz': 65421},
                #13
                {'batch_size': 1,
                 'x': 5,
                 'y': 200,
                 'z': 176,
                 'in_channels': 64,
                 'out_channels': 64,
                 'kernel_size': (3, 1, 1),
                 'strides': (2, 1, 1),
                 'paddings': (0, 0, 0),
                 'diff': 1e-3,
                 'nnz': 65421},
            ]

            i = 1
            values, indices = generate_data(config[i])

            p_shape = [config[i]['batch_size'], config[i]['x'],
                       config[i]['y'], config[i]['z'], config[i]['in_channels']]
            p_indices = paddle.to_tensor(indices, dtype='int32')
            p_indices = paddle.transpose(p_indices, perm=[1, 0])
            #p_values = paddle.to_tensor(values, dtype='float16')
            p_values = paddle.to_tensor(values)
            p_input = pi.sparse.sparse_coo_tensor(
                p_indices, p_values, p_shape, False)
            p_input = paddle.incubate.sparse.coalesce(p_input)
            p_conv = pi.sparse.nn.SubmConv3D(in_channels=config[i]['in_channels'], out_channels=config[i]['out_channels'], kernel_size=config[i]['kernel_size'],
                                          stride=config[i]['strides'], padding=config[i]['paddings'])

            c_out = p_conv(p_input,True)
            paddle.device.cuda.synchronize()
            p_out = p_conv(p_input)
            paddle.device.cuda.synchronize()
            p_out = paddle.incubate.sparse.coalesce(p_out)
            c_out = paddle.incubate.sparse.coalesce(c_out)
            assert np.array_equal(
                c_out.indices().numpy(), p_out.indices().numpy())
            assert np.allclose(c_out.values().numpy().flatten(),
                               p_out.values().numpy().flatten(), atol=config[i]['diff'], rtol=config[i]['diff'])

            paddle.device.cuda.synchronize()
            for n in range(100):
                c_out = p_conv(p_input,True)
            paddle.device.cuda.synchronize()

            t0 = time.perf_counter()
            paddle.device.cuda.synchronize()
            for n in range(500):
                c_out = p_conv(p_input, True)
            paddle.device.cuda.synchronize()
            t1 = time.perf_counter()

            paddle.device.cuda.synchronize()
            for n in range(100):
                p_out = p_conv(p_input)
            paddle.device.cuda.synchronize()

            t2 = time.perf_counter()
            paddle.device.cuda.synchronize()
            for n in range(500):
                p_out = p_conv(p_input)
            paddle.device.cuda.synchronize()
            t3 = time.perf_counter()

            #p_out = paddle.incubate.sparse.coalesce(p_out)


            #assert np.allclose(s_out.features.cpu().detach().numpy().flatten(),
                    #p_out.values().numpy().flatten(), atol=1e-3, rtol=1e-3)
            #print("spconv time:", t1-t0)
            print("cutlass time:", t1-t0)
            print("paddle time:", t3-t2)




