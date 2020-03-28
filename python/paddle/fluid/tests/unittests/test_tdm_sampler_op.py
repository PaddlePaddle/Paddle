#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid.layers as layers
import paddle.fluid as fluid
import random
import six


class TestTDMChildOp(OpTest):
    def setUp(self):
        self.__class__.op_type = "tdm_sampler"
        self.config()
        tree_travel = self.create_tdm_travel()
        tree_layer = self.create_tdm_layer()
        layer_node_num_list = [len(i) for i in tree_layer]
        layer_nums = 0
        node_nums = 0
        tree_layer_offset_lod = [0]
        tree_layer_flat = []
        for layer_idx, layer_node in enumerate(layer_node_num_list):
            layer_nums += 1
            node_nums += len(layer_node)
            tree_layer_flat += layer_node
            tree_layer_offset_lod.append(len(layer_node))

        travel_np = np.array(tree_travel).astype(self.dtype)
        layer_np = np.array(tree_layer_flat)
        layer_np = layer_np.reshape([-1, 1])

        x_np = np.random.randint(
            low=0, high=13, size=self.x_shape).astype(self.x_type)

        out_res = []
        label_res = []
        mask_res = []
        for batch in x_np:
            for node in batch:
                out_res += tree_travel[node]
                label = [1, 1, 1, 1]
                mask = [1, 1, 1, 1]
                if tree_travel[node][-1] == 0:
                    label[-1] = 0
                    mask[-1] = 0
                label_res += label
                mask_res += mask

        out_res_np = np.array(out_res).astype(self.x_type)
        label_res_np = np.array(label_res).astype(self.x_type)
        mask_res_np = np.array(mask_res).astype(self.x_type)

        out = np.reshape(out_res_np, self.output_shape)
        label = np.reshape(label_res_np, self.output_shape)
        mask = np.reshape(mask_res_np, self.output_shape)

        self.attrs = {
            'neg_samples_num_list': self.neg_samples_num_list,
            'output_positive': True,
            'layer_offset_lod': tree_layer_offset_lod,
            'seed': 0
        }
        self.inputs = {'X': x_np, 'Travel': travel_np, 'Layer': layer_np}
        self.outputs = {'Out': child, 'Labels': label, 'Mask': mask}

    def create_tdm_travel(self):
        tree_travel = [[1, 3, 7, 14], [1, 3, 7, 15], [1, 3, 8, 16],
                       [1, 3, 8, 17], [1, 4, 9, 18], [1, 4, 9, 19],
                       [1, 4, 10, 20], [1, 4, 10, 21], [2, 5, 11, 22],
                       [2, 5, 11, 23], [2, 5, 12, 24], [2, 5, 12, 25],
                       [2, 6, 13, 0]]
        return tree_travel

    def create_tdm_layer(self):
        tree_layer = [[1, 2], [3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13],
                      [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]]
        return tree_layer

    def config(self):
        """set test shape & type"""
        self.neg_samples_num_list = [0, 0, 0, 0]
        self.x_shape = (10, 1)
        self.output_shape = (10, 4)
        self.x_type = 'int32'
        self.dtype = 'int32'

    def test_check_output(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
