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

def generate_data(config):
    values = []
    indices = []
    nnz = int(config['batch_size'] * config['x'] * config['y'] * config['z'] * (1-config['sparsity']))
    print(nnz)

    for i in range(nnz):
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
                #{'batch_size': 8,
                 #'x': 41,
                 #'y': 250,
                 #'z': 250,
                 #'kernel_size': (3, 3, 3),
                 #'in_channels': 3,
                 #'out_channels': 15,
                 #'paddings': (0, 0, 0),
                 #'strides': (1, 1, 1),
                 #'dilations': (1, 1, 1),
                 #'diff': 1e-3,
                 #'sparsity': 0.99},
                #{'batch_size': 8,
                 #'x': 41,
                 #'y': 250,
                 #'z': 250,
                 #'kernel_size': (3, 3, 3),
                 #'in_channels': 3,
                 #'out_channels': 16,
                 #'paddings': (0, 0, 0),
                 #'strides': (1, 1, 1),
                 #'dilations': (1, 1, 1),
                 #'diff': 1e-3,
                 #'sparsity': 0.99},
                #{'batch_size': 8,
                 #'x': 41,
                 #'y': 250,
                 #'z': 250,
                 #'kernel_size': (3, 3, 3),
                 #'in_channels': 3,
                 #'out_channels': 17,
                 #'paddings': (0, 0, 0),
                 #'strides': (1, 1, 1),
                 #'dilations': (1, 1, 1),
                 #'diff': 1e-3,
                 #'sparsity': 0.99},
                #{'batch_size': 8,
                 #'x': 41,
                 #'y': 250,
                 #'z': 250,
                 #'kernel_size': (3, 3, 3),
                 #'in_channels': 4,
                 #'out_channels': 15,
                 #'paddings': (0, 0, 0),
                 #'strides': (1, 1, 1),
                 #'dilations': (1, 1, 1),
                 #'diff': 1e-3,
                 #'sparsity': 0.99},
                #{'batch_size': 8,
                 #'x': 41,
                 #'y': 250,
                 #'z': 250,
                 #'kernel_size': (3, 3, 3),
                 #'in_channels': 4,
                 #'out_channels': 16,
                 #'paddings': (0, 0, 0),
                 #'strides': (1, 1, 1),
                 #'dilations': (1, 1, 1),
                 #'diff': 1e-3,
                 #'sparsity': 0.99},
                #{'batch_size': 8,
                 #'x': 41,
                 #'y': 250,
                 #'z': 250,
                 #'kernel_size': (3, 3, 3),
                 #'in_channels': 4,
                 #'out_channels': 17,
                 #'paddings': (0, 0, 0),
                 #'strides': (1, 1, 1),
                 #'dilations': (1, 1, 1),
                 #'diff': 1e-3,
                 #'sparsity': 0.99},
                #{'batch_size': 8,
                 #'x': 41,
                 #'y': 250,
                 #'z': 250,
                 #'kernel_size': (3, 3, 3),
                 #'in_channels': 5,
                 #'out_channels': 15,
                 #'paddings': (0, 0, 0),
                 #'strides': (1, 1, 1),
                 #'dilations': (1, 1, 1),
                 #'diff': 1e-3,
                 #'sparsity': 0.99},
                #{'batch_size': 8,
                 #'x': 41,
                 #'y': 250,
                 #'z': 250,
                 #'kernel_size': (3, 3, 3),
                 #'in_channels': 5,
                 #'out_channels': 16,
                 #'paddings': (0, 0, 0),
                 #'strides': (1, 1, 1),
                 #'dilations': (1, 1, 1),
                 #'diff': 1e-3,
                 #'sparsity': 0.99},
                #{'batch_size': 8,
                 #'x': 41,
                 #'y': 250,
                 #'z': 250,
                 #'kernel_size': (3, 3, 3),
                 #'in_channels': 5,
                 #'out_channels': 17,
                 #'paddings': (0, 0, 0),
                 #'strides': (1, 1, 1),
                 #'dilations': (1, 1, 1),
                 #'diff': 1e-3,
                 #'sparsity': 0.99},
                {'batch_size': 8,
                 'x': 41,
                 'y': 250,
                 'z': 250,
                 'kernel_size': (3, 3, 3),
                 'in_channels': 4,
                 'out_channels': 16,
                 'paddings': (0, 0, 0),
                 'strides': (1, 1, 1),
                 'dilations': (1, 1, 1),
                 'diff': 1e-3,
                 'sparsity': 0.99}
            ]

            for i in range(len(config)):
                values, indices = generate_data(config[i])

                p_shape = [config[i]['batch_size'], config[i]['x'],
                        config[i]['y'], config[i]['z'], config[i]['in_channels']]
                p_indices = paddle.to_tensor(indices, dtype='int32')
                p_indices = paddle.transpose(p_indices, perm=[1, 0])
                p_values = paddle.to_tensor(values, dtype='float32')
                p_input = pi.sparse.sparse_coo_tensor(p_indices, p_values, p_shape, False)
                p_input = paddle.incubate.sparse.coalesce(p_input)
                p_conv = pi.sparse.nn.Conv3D(in_channels=config[i]['in_channels'], out_channels=config[i]['out_channels'], kernel_size=config[i]['kernel_size'],
                                            stride=config[i]['strides'], padding=config[i]['paddings'], dilation=config[i]['dilations'])

                #device = torch.device("cuda")
                #spatial_shape = [config[i]['x'], config[i]['y'], config[i]['z']]
                #s_values = torch.tensor(np.array(p_input.values()), device=device)
                #s_indices = torch.tensor(np.array(paddle.transpose(p_input.indices(),perm=[1,0])), device=device).int()
                #s_input = spconv.SparseConvTensor(
                    #s_values, s_indices, spatial_shape, config[i]['batch_size'])
                #s_conv = spconv.SparseConv3d(config[i]['in_channels'], config[i]['out_channels'], kernel_size=config[i]['kernel_size'],
                                            #stride=config[i]['strides'], padding=config[i]['paddings'], dilation=config[i]['dilations'], bias=False)

                #s_conv.weight = torch.nn.Parameter(torch.tensor(
                    #np.transpose(p_conv.weight.numpy(), (4, 0, 1, 2, 3))).cuda().contiguous())

                print(i)
                #torch.cuda.synchronize(device=device)
                #t0 = time.time()
                #s_out = s_conv(s_input)
                #torch.cuda.synchronize(device=device)
                #t1 = time.time()

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

                #assert np.array_equal(
                    #s_out.indices.cpu().detach().numpy().transpose(1, 0), p_out.indices().numpy())

                #assert np.allclose(s_out.features.cpu().detach().numpy().flatten(),
                                #p_out.values().numpy().flatten(), atol=1e-3, rtol=1e-3)
                #print("spconv time:", t1-t0)
                print("paddle time:", t3-t2)




