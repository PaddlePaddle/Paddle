# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class TestLabelSmoothSPMDRule(unittest.TestCase):
    def setUp(self):
        label_shape = [16, 16, 16]
        out_shape = [16, 16, 16]
        prior_dist_shape = [1, 16]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])

        label_tensor_dist_attr = TensorDistAttr()
        label_tensor_dist_attr.dims_mapping = [-1, -1, -1]
        label_tensor_dist_attr.process_mesh = process_mesh

        self.label_dist_tensor_spec = DistTensorSpec(
            label_shape, label_tensor_dist_attr
        )
        prior_dist_tensor_dist_attr = TensorDistAttr()
        prior_dist_tensor_dist_attr.dims_mapping = [-1, -1]
        prior_dist_tensor_dist_attr.process_mesh = process_mesh

        self.prior_dist_dist_tensor_spec = DistTensorSpec(
            prior_dist_shape, prior_dist_tensor_dist_attr
        )

        out_tensor_dist_attr = TensorDistAttr()
        out_tensor_dist_attr.dims_mapping = [-1, -1, -1]
        out_tensor_dist_attr.process_mesh = process_mesh
        self.out_dist_tensor_spec = DistTensorSpec(
            out_shape, out_tensor_dist_attr
        )

        self.rule = core.get_phi_spmd_rule("label_smooth")
        self.attrs = OrderedDict()
        self.attrs['epsilon'] = 0.1

    def test_label_smooth_forward(self):
        # [0, 1, -1], [Fake] --> [0, 1, -1], [Fake], [0, 1, -1]
        self.label_dist_tensor_spec.set_dims_mapping([0, 1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.label_dist_tensor_spec,
            DistTensorSpec(),
            self.attrs['epsilon'],
        )

        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_input_dist_attrs[1], TensorDistAttr())
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])

        # shape: [16, 16, 16], [1, 16]. [0, 1, -1], [-1, -1] --> [0, 1, -1], [-1, -1], [0, 1, -1]
        self.prior_dist_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.label_dist_tensor_spec,
            self.prior_dist_dist_tensor_spec,
            self.attrs['epsilon'],
        )

        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])

        # shape: [16, 16, 16], [1, 16]. [0, 1, -1], [-1, 1] --> [0, 1, -1], [-1, -1], [0, 1, -1]
        self.prior_dist_dist_tensor_spec.set_dims_mapping([-1, 1])
        result_dist_attrs = self.rule.infer_forward(
            self.label_dist_tensor_spec,
            self.prior_dist_dist_tensor_spec,
            self.attrs['epsilon'],
        )
        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])

        # shape: [16, 16, 16], [1, 16]. [0, 1, -1], [-1, 0] --> [0, 1, -1], [-1, -1], [0, 1, -1]
        self.prior_dist_dist_tensor_spec.set_dims_mapping([-1, 0])
        result_dist_attrs = self.rule.infer_forward(
            self.label_dist_tensor_spec,
            self.prior_dist_dist_tensor_spec,
            self.attrs['epsilon'],
        )
        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])

        # shape: [16, 16, 16], [1, 16]. [0, -1, 1], [-1, 1] --> [0, -1, 1], [-1, 1], [0, -1, 1]
        self.label_dist_tensor_spec.set_dims_mapping([0, -1, 1])
        self.prior_dist_dist_tensor_spec.set_dims_mapping([-1, 1])
        result_dist_attrs = self.rule.infer_forward(
            self.label_dist_tensor_spec,
            self.prior_dist_dist_tensor_spec,
            self.attrs['epsilon'],
        )
        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, 1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1, 1])

        # shape: [16, 16, 16], [1, 16]. [0, -1, 1], [-1, 0] --> [0, -1, 1], [-1, 1], [0, -1, 1]
        self.prior_dist_dist_tensor_spec.set_dims_mapping([-1, 0])
        result_dist_attrs = self.rule.infer_forward(
            self.label_dist_tensor_spec,
            self.prior_dist_dist_tensor_spec,
            self.attrs['epsilon'],
        )
        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, 1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1, 1])

        # shape: [16, 16, 16], [1, 16]. [0, -1, 1], [-1, -1] --> [0, -1, 1], [-1, 1], [0, -1, 1]
        self.prior_dist_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.label_dist_tensor_spec,
            self.prior_dist_dist_tensor_spec,
            self.attrs['epsilon'],
        )
        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, 1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1, 1])

        # shape: [16, 16, 16], [16, 16, 16]. [0, 1, -1], [0, -1, 1] --> [0, 1, -1], [0, 1, -1], [0, 1, -1]
        self.label_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        self.prior_dist_dist_tensor_spec.shape = [16, 16, 16]
        self.prior_dist_dist_tensor_spec.set_dims_mapping([0, -1, 1])
        result_dist_attrs = self.rule.infer_forward(
            self.label_dist_tensor_spec,
            self.prior_dist_dist_tensor_spec,
            self.attrs['epsilon'],
        )
        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])

    def test_label_smooth_backward(self):
        # shape: [16, 16 ,16]. [0, -1, 1] --> [0, -1, 1], [0, -1, 1]
        self.out_dist_tensor_spec.set_dims_mapping([0, -1, 1])
        result_dist_attrs = self.rule.infer_backward(
            self.out_dist_tensor_spec,
            self.attrs['epsilon'],
        )

        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, 1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1, 1])


if __name__ == "__main__":
    unittest.main()
