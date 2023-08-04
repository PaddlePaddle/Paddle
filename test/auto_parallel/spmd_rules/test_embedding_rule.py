# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

from paddle.distributed.auto_parallel.static.completion import get_spmd_rule
from paddle.distributed.auto_parallel.static.dist_attribute import (
    DistTensorSpec,
    TensorDistAttr,
)
from paddle.distributed.fleet import auto


class TestEmbeddingSPMDRule(unittest.TestCase):
    def setUp(self):
        self.rule1 = get_spmd_rule("lookup_table_v2")

        x_shape = [4, 1024]  # [B,S]
        table_shape = [512, 768]  # [V,H]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        table_tensor_dist_attr = TensorDistAttr()
        table_tensor_dist_attr.process_mesh = process_mesh
        self.table_dist_tensor_spec = DistTensorSpec(
            table_shape, table_tensor_dist_attr
        )

        self.attrs = {
            'padding_idx': -1,
            'sparse': False,
        }

    def test_embedding_infer_forward(self):
        # data parallel
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.table_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule1.infer_forward(
            [self.x_dist_tensor_spec, self.table_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 2)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, -1])

        # table col-wise parallel & dp
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.table_dist_tensor_spec.set_dims_mapping([-1, 0])
        result_dist_attrs = self.rule1.infer_forward(
            [self.x_dist_tensor_spec, self.table_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, 0])

        # table row-wise parallel & dp
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.table_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule1.infer_forward(
            [self.x_dist_tensor_spec, self.table_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, -1])

        # table row-wise parallel & padding_idx
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.table_dist_tensor_spec.set_dims_mapping([0, -1])
        self.attrs['padding_idx'] = 128
        with self.assertRaises(ValueError):
            result_dist_attrs = self.rule1.infer_forward(
                [self.x_dist_tensor_spec, self.table_dist_tensor_spec],
                self.attrs,
            )

        # table row-wise parallel & sparse
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.table_dist_tensor_spec.set_dims_mapping([0, -1])
        self.attrs['padding_idx'] = -1
        self.attrs['sparse'] = True
        with self.assertRaises(ValueError):
            result_dist_attrs = self.rule1.infer_forward(
                [self.x_dist_tensor_spec, self.table_dist_tensor_spec],
                self.attrs,
            )


if __name__ == "__main__":
    unittest.main()
