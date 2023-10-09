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

from paddle.distributed.auto_parallel.static.dist_attribute import (
    DistTensorSpec,
    TensorDistAttr,
)
from paddle.distributed.fleet import auto
from paddle.framework import core


class TestReplicatedSPMDRule(unittest.TestCase):
    def setUp(self):
        # After replaced all spmd rules by phi impl, we can recover the
        # api name to `get_spmd_rule`
        self.rule = core.get_phi_spmd_rule("replicated")

        x_shape = [10, 10, 32, 48]
        y_shape = [32, 48]
        out1_shape = [10, 10, 32, 48]
        out2_shape = [10, 32, 48]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [-1, 1, -1, -1]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        y_tensor_dist_attr = TensorDistAttr()
        y_tensor_dist_attr.dims_mapping = [0, -1]
        y_tensor_dist_attr.process_mesh = process_mesh
        self.y_dist_tensor_spec = DistTensorSpec(y_shape, y_tensor_dist_attr)

        out1_tensor_dist_attr = TensorDistAttr()
        # out1_tensor_dist_attr.dims_mapping = [-1, 1, 0, -1] // unset on purpose, inferforward only need shape infer of output
        # out1_tensor_dist_attr.process_mesh = process_mesh
        self.out1_dist_tensor_spec = DistTensorSpec(
            out1_shape, out1_tensor_dist_attr
        )
        out2_tensor_dist_attr = TensorDistAttr()
        self.out2_dist_tensor_spec = DistTensorSpec(
            out2_shape, out2_tensor_dist_attr
        )

    def test_replicated_infer_forward(self):
        # return all -1
        # 2 inputs 2 outputs
        in_vec = [self.x_dist_tensor_spec, self.y_dist_tensor_spec]
        out_vec = [self.out1_dist_tensor_spec, self.out2_dist_tensor_spec]

        result_dist_attrs = self.rule.infer_forward(in_vec, out_vec)

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(result_dist_attrs[0]), 2)
        self.assertEqual(len(result_dist_attrs[1]), 2)

        self.assertEqual(result_dist_attrs[0][0].dims_mapping, [-1, -1, -1, -1])
        self.assertEqual(result_dist_attrs[0][1].dims_mapping, [-1, -1])
        self.assertEqual(result_dist_attrs[1][0].dims_mapping, [-1, -1, -1, -1])
        self.assertEqual(result_dist_attrs[1][1].dims_mapping, [-1, -1, -1])

        # 1 inputs 2 outputs
        in_vec = [self.y_dist_tensor_spec]
        out_vec = [self.out1_dist_tensor_spec, self.out2_dist_tensor_spec]

        result_dist_attrs = self.rule.infer_forward(in_vec, out_vec)

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(result_dist_attrs[0]), 1)
        self.assertEqual(len(result_dist_attrs[1]), 2)

        self.assertEqual(result_dist_attrs[0][0].dims_mapping, [-1, -1])
        self.assertEqual(result_dist_attrs[1][0].dims_mapping, [-1, -1, -1, -1])
        self.assertEqual(result_dist_attrs[1][1].dims_mapping, [-1, -1, -1])

    def test_replicated_infer_backward(self):
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        self.out1_dist_tensor_spec.set_dims_mapping([-1, 1, 0, -1])
        self.out1_dist_tensor_spec.set_process_mesh(process_mesh)
        self.out2_dist_tensor_spec.set_dims_mapping([1, -1, 0])
        self.out2_dist_tensor_spec.set_process_mesh(process_mesh)

        in_vec = [self.x_dist_tensor_spec, self.y_dist_tensor_spec]
        out_vec = [self.out1_dist_tensor_spec, self.out2_dist_tensor_spec]

        result_dist_attrs = self.rule.infer_backward(in_vec, out_vec)

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(result_dist_attrs[0]), 2)
        self.assertEqual(len(result_dist_attrs[1]), 2)

        self.assertEqual(result_dist_attrs[0][0].dims_mapping, [-1, -1, -1, -1])
        self.assertEqual(result_dist_attrs[0][1].dims_mapping, [-1, -1])
        self.assertEqual(result_dist_attrs[1][0].dims_mapping, [-1, -1, -1, -1])
        self.assertEqual(result_dist_attrs[1][1].dims_mapping, [-1, -1, -1])

        # 1 inputs 3 outputs
        in_vec = [self.y_dist_tensor_spec]
        out_vec = [
            self.x_dist_tensor_spec,
            self.out1_dist_tensor_spec,
            self.out2_dist_tensor_spec,
        ]

        result_dist_attrs = self.rule.infer_backward(in_vec, out_vec)

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(result_dist_attrs[0]), 1)
        self.assertEqual(len(result_dist_attrs[1]), 3)

        self.assertEqual(result_dist_attrs[0][0].dims_mapping, [-1, -1])
        self.assertEqual(result_dist_attrs[1][0].dims_mapping, [-1, -1, -1, -1])
        self.assertEqual(result_dist_attrs[1][1].dims_mapping, [-1, -1, -1, -1])
        self.assertEqual(result_dist_attrs[1][2].dims_mapping, [-1, -1, -1])


if __name__ == "__main__":
    unittest.main()
