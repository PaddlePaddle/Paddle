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


class TestReductionSPMDRule(unittest.TestCase):
    """
    Unit tests for split spmd rule.
    """

    def setUp(self):
        self.rule = get_spmd_rule("split")

        x_shape = [64, 32, 48]
        process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [1, 0]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        self.attrs = {
            'num_or_sections': 2,
            'axis': 1,
        }

    def test_single_mesh_dim(self):
        # num_or_sections = 2, axis = 1
        # [0, -1, -1] --> [0, -1, -1], [0, -1, -1], [0, -1, -1]
        self.attrs['num_or_sections'] = 2
        self.attrs['axis'] = 1
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 2)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [0, -1, -1])

        # num_or_sections = [15, 16, 17], axis = 2
        # [0, -1, -1] --> [0, -1, -1], [0, -1, -1], [0, -1, -1], [0, -1, -1]
        self.attrs['num_or_sections'] = [15, 16, 17]
        self.attrs['axis'] = 2
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 3)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[2].dims_mapping, [0, -1, -1])

        # num_or_sections = [15, 16, 17], axis = 2
        # [-1, -1, 0] --> [-1, -1, -1], [-1, -1, -1], [-1 -1, -1], [-1, -1, -1]
        self.attrs['num_or_sections'] = [15, 16, 17]
        self.attrs['axis'] = 2
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 3)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1])
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1]
        )
        self.assertEqual(
            infered_output_dist_attrs[1].dims_mapping, [-1, -1, -1]
        )
        self.assertEqual(
            infered_output_dist_attrs[2].dims_mapping, [-1, -1, -1]
        )

        # num_or_sections = 2, axis = -2
        # [0, -1, -1] --> [0, -1, -1], [0, -1, -1], [0, -1, -1]
        self.attrs['num_or_sections'] = 2
        self.attrs['axis'] = -2
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 2)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [0, -1, -1])

    def test_multi_mesh_dim(self):
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        self.x_dist_tensor_spec.set_process_mesh(process_mesh)
        self.x_dist_tensor_spec.shape = [96, 32, 48, 24]

        # num_or_sections = 3, axis = -1
        # [0, 1, -1, -1] --> [0, 1, -1, -1], [0, 1, -1, -1], [0, 1, -1, -1], [0, 1, -1, -1]
        self.attrs['num_or_sections'] = 3
        self.attrs['axis'] = -1
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1, -1])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 3)

        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )
        self.assertEqual(
            infered_output_dist_attrs[1].dims_mapping, [0, 1, -1, -1]
        )
        self.assertEqual(
            infered_output_dist_attrs[2].dims_mapping, [0, 1, -1, -1]
        )

        # num_or_sections = [32, 32, 32], axis = 0
        # [0, 1, -1, -1] --> [-1, 1, -1, -1], [-1, 1, -1, -1], [-1, 1, -1, -1], [-1, 1, -1, -1]
        self.attrs['num_or_sections'] = [32, 32, 32]
        self.attrs['axis'] = 0
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1, -1])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 3)

        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [-1, 1, -1, -1]
        )
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, 1, -1, -1]
        )
        self.assertEqual(
            infered_output_dist_attrs[1].dims_mapping, [-1, 1, -1, -1]
        )
        self.assertEqual(
            infered_output_dist_attrs[2].dims_mapping, [-1, 1, -1, -1]
        )


if __name__ == "__main__":
    unittest.main()
