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


class TestMatmulSPMDRule(unittest.TestCase):
    def setUp(self):
        self.rule = get_spmd_rule("replicated")

        x_shape = [64, 32, 10, 10]
        y_shape = [32, 48]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [-1, 1, 0, -1]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        y_tensor_dist_attr = TensorDistAttr()
        y_tensor_dist_attr.dims_mapping = [0, -1]
        y_tensor_dist_attr.process_mesh = process_mesh
        self.y_dist_tensor_spec = DistTensorSpec(y_shape, y_tensor_dist_attr)

    def test_replicated_infer_forward(self):
        # return all -1
        result_tensor_specs = self.rule.infer_forward(
            [self.x_dist_tensor_spec, self.y_dist_tensor_spec], {}
        )
        self.assertEqual(len(result_tensor_specs), 2)
        self.assertEqual(len(result_tensor_specs[0]), 2)
        self.assertEqual(len(result_tensor_specs[1]), 1)
        self.assertEqual(
            result_tensor_specs[0][0].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(result_tensor_specs[0][1].dims_mapping, [-1, -1])
        self.assertEqual(
            result_tensor_specs[1][0].dims_mapping, [-1, -1, -1, -1]
        )


if __name__ == "__main__":
    unittest.main()
