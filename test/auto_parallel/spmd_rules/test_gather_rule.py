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
from collections import OrderedDict

from paddle.distributed.auto_parallel.static.dist_attribute import (
    DistTensorSpec,
    TensorDistAttr,
)
from paddle.distributed.fleet import auto
from paddle.framework import core


class TestScatterSPMDRule(unittest.TestCase):
    """
    Unit tests for scatter spmd rule.
    """

    def setUp(self):
        x_shape = [64, 32, 48]
        index_shape = [16]
        updates_shape = [32, 32, 48]
        process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3])
        self.attrs = OrderedDict()
        self.attrs['axis'] = 0
        self.rule = core.get_phi_spmd_rule("gather")

        x_dist_attr = TensorDistAttr()
        x_dist_attr.dims_mapping = [-1, -1, -1]
        x_dist_attr.process_mesh = process_mesh
        self.x_spec = DistTensorSpec(x_shape, x_dist_attr)

        index_dist_attr = TensorDistAttr()
        index_dist_attr.dims_mapping = [-1]
        index_dist_attr.process_mesh = process_mesh
        self.index_spec = DistTensorSpec(index_shape, index_dist_attr)

    def test_single_mesh_dim(self):
        # axis: 0
        # dims_mapping: [0, -1, -1], [-1] --> [-1, -1, -1], [-1], [-1, -1, -1]
        self.attrs['axis'] = 0
        self.x_spec.set_dims_mapping([0, -1, -1])
        self.index_spec.set_dims_mapping([-1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_spec,
            self.index_spec,
            self.attrs['axis'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 2)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1]
        )

        # axis: 0
        # dims_mapping: [-1, 0, -1], [-1] --> [-1, 0, -1], [-1], [-1, 0, -1]
        self.attrs['axis'] = 0
        self.x_spec.set_dims_mapping([-1, 0, -1])
        self.index_spec.set_dims_mapping([-1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_spec,
            self.index_spec,
            self.attrs['axis'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 2)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, 0, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0, -1])

        # axis: 0
        # dims_mapping: [0, -1, -1], [0] --> [-1, -1, -1], [0], [0, -1, -1]
        self.attrs['axis'] = 0
        self.x_spec.set_dims_mapping([0, -1, -1])
        self.index_spec.set_dims_mapping([0])
        result_dist_attrs = self.rule.infer_forward(
            self.x_spec,
            self.index_spec,
            self.attrs['axis'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 2)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1, -1])

        # 0-d tensor
        # axis: 1
        # dims_mapping: [-1, 0, -1], [0] --> [-1, -1, -1], [-1], [-1, -1]
        self.attrs['axis'] = 1
        self.index_spec.shape = []
        self.x_spec.set_dims_mapping([-1, 0, -1])
        self.index_spec.set_dims_mapping([0])

        result_dist_attrs = self.rule.infer_forward(
            self.x_spec,
            self.index_spec,
            self.attrs['axis'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 2)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1])
        self.index_spec.shape = [16]

    def test_multi_mesh_dim(self):
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        self.x_spec.set_process_mesh(process_mesh)
        self.index_spec.set_process_mesh(process_mesh)

        # axis = 1
        # [0, 1, -1], [1] --> [0, -1, -1], [1], [0, 1, -1]
        self.attrs['axis'] = 1
        self.x_spec.set_dims_mapping([0, 1, -1])
        self.index_spec.set_dims_mapping([1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_spec,
            self.index_spec,
            self.attrs['axis'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 2)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, 1, -1])

        # [0, 1, -1], [0] --> [0, -1, -1], [0], [0, -1, -1]
        self.attrs['axis'] = 1
        self.x_spec.set_dims_mapping([0, 1, -1])
        self.index_spec.set_dims_mapping([0])
        result_dist_attrs = self.rule.infer_forward(
            self.x_spec,
            self.index_spec,
            self.attrs['axis'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 2)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1, -1])

    def test_reverse_multi_mesh_dim(self):
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        self.x_spec.set_process_mesh(process_mesh)
        self.index_spec.set_process_mesh(process_mesh)
        self.out_spec = DistTensorSpec(self.x_spec)

        # axis = 1
        # [1, 0, -1] --> [1, -1, -1], [0], [1, 0, -1]
        self.attrs['axis'] = 1
        self.out_spec.set_dims_mapping([1, 0, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_spec,
            self.index_spec,
            self.out_spec,
            self.attrs['axis'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 2)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, 0, -1])


if __name__ == "__main__":
    unittest.main()
