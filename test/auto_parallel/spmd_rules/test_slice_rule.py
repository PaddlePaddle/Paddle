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


class TestSliceSPMDRule(unittest.TestCase):
    """
    Unit tests for slice spmd rule.
    """

    def setUp(self):
        input_shape = [64, 32, 48]
        process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3])

        input_tensor_dist_attr = TensorDistAttr()
        input_tensor_dist_attr.dims_mapping = [1, 0]
        input_tensor_dist_attr.process_mesh = process_mesh
        self.input_dist_tensor_spec = DistTensorSpec(
            input_shape, input_tensor_dist_attr
        )

    def test_single_mesh_dim(self):
        # axes = [1, 2], starts = [8, 16], ends = [16, 32]
        # [-1, -1, 0] --> [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]
        self.rule = core.get_phi_spmd_rule("slice")
        self.attrs = OrderedDict()
        self.attrs['axes'] = [1, 2]
        self.attrs['starts'] = [8, 16]
        self.attrs['ends'] = [16, 32]
        self.attrs['infer_flags'] = []
        self.attrs['decrease_axis'] = []
        self.input_dist_tensor_spec.set_dims_mapping([-1, -1, 0])
        result_dist_attrs = self.rule.infer_forward(
            self.input_dist_tensor_spec,
            self.attrs['axes'],
            self.attrs['starts'],
            self.attrs['ends'],
            self.attrs['infer_flags'],
            self.attrs['decrease_axis'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1])
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1]
        )


if __name__ == "__main__":
    unittest.main()
