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


class TestTDMSamplerOp(OpTest):
    def setUp(self):
        self.__class__.op_type = "tdm_sampler"
        self.config()

        self.tree_travel = self.create_tdm_travel()
        self.tree_layer = self.create_tdm_layer()

        output_0 = self.x_shape[0]
        output_1 = len(self.neg_samples_num_list) + \
            np.sum(self.neg_samples_num_list)
        self.output_shape = (output_0, output_1)
        self.layer_sample_nums = [1 + i for i in self.neg_samples_num_list]

        layer_node_num_list = [len(i) for i in self.tree_layer]
        tree_layer_offset_lod = [0]
        tree_layer_flat = []
        node_nums = 0
        for layer_idx, layer_node in enumerate(layer_node_num_list):
            tree_layer_flat += self.tree_layer[layer_idx]
            node_nums += layer_node
            tree_layer_offset_lod.append(node_nums)

        travel_np = np.array(self.tree_travel).astype(self.dtype)
        layer_np = np.array(tree_layer_flat).astype(self.dtype)
        layer_np = layer_np.reshape([-1, 1])

        x_np = np.random.randint(
            low=0, high=13, size=self.x_shape).astype(self.x_type)

        out = np.random.random(self.output_shape).astype(self.x_type)
        label = np.random.random(self.output_shape).astype(self.x_type)
        mask = np.random.random(self.output_shape).astype(self.x_type)

        self.attrs = {
            'neg_samples_num_list': self.neg_samples_num_list,
            'output_positive': True,
            'layer_offset_lod': tree_layer_offset_lod,
            'seed': 0
        }
        self.inputs = {'X': x_np, 'Travel': travel_np, 'Layer': layer_np}
        self.outputs = {'Out': out, 'Labels': label, 'Mask': mask}

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
        self.x_type = 'int32'
        self.dtype = 'int32'

    def test_check_output(self):
        self.check_output_customized(self.verify_output)
        x_res, label_res, mask_res = self.out

        # check dtype
        if self.x_type == 'int32':
            assert x_res.dtype == np.int32
            assert label_res.dtype == np.int32
            assert mask_res.dtype == np.int32
        elif self.x_type == 'int64':
            assert x_res.dtype == np.int64
            assert label_res.dtype == np.int64
            assert mask_res.dtype == np.int64

        x_res = x_res.reshape(output_shape)
        label_res = label_res.reshape(output_shape)
        mask_res = mask_res.reshape(output_shape)

        layer_nums = len(self.neg_samples_num_list)
        for batch_ids, x_batch in enumerate(x_res):
            start_offset = 0
            for layer_idx in range(layer_nums):
                end_offset = start_offset + self.layer_sample_nums[layer_idx]
                sampling_res = x_batch[start_offset:end_offset]
                # check unique
                assert (np.unique(sampling_res)).shape == sampling_res.sahpe
                # check legal
                assert np.isin(sampling_res,
                               np.array(self.tree_layer[layer_idx])).all()
                label_sampling_res = label_res[batch_ids][start_offset:
                                                          end_offset]
                mask_sampling_res = mask_res[batch_ids][start_offset:end_offset]
                # check label
                assert label_sampling_res[0] == 1
                assert np.sum(label_sampling_res) == 1
                # check mask
                padding_index = np.where(sampling_res == 0)
                assert np.sum(mask_sampling_res[padding_index]) == 0

    def verify_output(self, outs):
        self.out = outs


if __name__ == "__main__":
    unittest.main()
