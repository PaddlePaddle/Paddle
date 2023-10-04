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


class TestSqueezeSPMDRule(unittest.TestCase):
    def setUp(self):
        self.rule = core.get_phi_spmd_rule("squeeze")

        x_shape = [1, 4, 1, 16]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [-1, -1, -1, -1]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

    def test_squeeze_infer_forward(self):
        # shape: [1, 4, 1, 16] --> [1, 4, 16]
        # dims_mapping: [0, 1, -1, -1] --> [0, 1, -1, -1] [0, 1, -1]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1, -1])
        self.attrs = OrderedDict()
        self.attrs['axis'] = [2]
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, 1, -1])


if __name__ == "__main__":
    unittest.main()
