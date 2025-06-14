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


class TestExpandAsSPMDRule(unittest.TestCase):
    def setUp(self):
        x_shape = [1, 48]
        y_shape = [2, 32, 48]
        out_shape = [2, 32, 48]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [-1, -1]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        y_tensor_dist_attr = TensorDistAttr()
        y_tensor_dist_attr.dims_mapping = [0, 1, -1]
        y_tensor_dist_attr.process_mesh = process_mesh
        self.y_dist_tensor_spec = DistTensorSpec(y_shape, y_tensor_dist_attr)
        out_tensor_dist_attr = TensorDistAttr()
        out_tensor_dist_attr.dims_mapping = [0, 1, -1]
        out_tensor_dist_attr.process_mesh = process_mesh
        self.out_dist_tensor_spec = DistTensorSpec(
            out_shape, out_tensor_dist_attr
        )

        self.rule = core.get_phi_spmd_rule("expand_as")
        self.attrs = OrderedDict()
        self.attrs['target_shape'] = [2, 32, 48]

    def test_expand_as_forward_with_y(self):
        # [1, 48], [2, 32, 48]
        # [-1, -1], [0, 1, -1] --> [-1, -1], [0, 1, -1],[0, 1, -1]
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.attrs['target_shape'],
        )

        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])

        # [1, 48], [2, 32, 48]
        # [-1, -1], [0, -1, 1] --> [-1, 1], [0, -1, 1], [0, -1, 1]
        self.y_dist_tensor_spec.set_dims_mapping([0, -1, 1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.attrs['target_shape'],
        )

        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1, 1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, -1, 1])

        # [2, 1, 48], [2, 32, 48]
        # [0, -1, -1], [0, 1, -1] --> [0, -1, -1], [0, 1, -1], [0, 1, -1]
        self.x_dist_tensor_spec.shape = [2, 1, 48]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.attrs['target_shape'],
        )
        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])

        # [2, 1, 48], [2, 32, 48]
        # [-1, -1, 0], [0, 1, -1] --> [0, -1, -1], [0, 1, -1], [0, 1, -1]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0])
        self.y_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.attrs['target_shape'],
        )
        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])

    def test_expand_as_forward_without_y(self):
        # [1, 48], target =[2, 32, 48]
        # [-1, -1] --> [-1, -1], [-1, -1, -1]
        self.attrs['target_shape'] = [2, 32, 48]
        self.x_dist_tensor_spec.shape = [1, 48]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            DistTensorSpec(),
            self.attrs['target_shape'],
        )
        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[1], TensorDistAttr())
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, -1]
        )

        # [1, 48], target =[2, 32, 48]
        # [-1, 0] --> [-1, 0], [-1, -1, 0]
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            DistTensorSpec(),
            self.attrs['target_shape'],
        )
        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 0])
        self.assertEqual(inferred_input_dist_attrs[1], TensorDistAttr())
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, 0]
        )

        # [2, 1, 48], target =[2, 32, 48]
        # [0, -1, -1] --> [0, -1, -1], [0, -1, -1]
        self.x_dist_tensor_spec.shape = [2, 1, 48]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            DistTensorSpec(),
            self.attrs['target_shape'],
        )
        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(inferred_input_dist_attrs[1], TensorDistAttr())
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1]
        )

        # [2, 32, 48], target =[2, 32, 48]
        # [0, 1, -1] --> [0, 1, -1], [0, 1, -1]
        self.x_dist_tensor_spec.shape = [2, 32, 48]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            DistTensorSpec(),
            self.attrs['target_shape'],
        )
        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_input_dist_attrs[1], TensorDistAttr())
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1])

    def test_expand_as_backward(self):
        # [1, 48], [2, 32, 48]
        # [-1, -1], [0, 1, -1] --> [-1, -1], [0, 1, -1], [-1, -1](partial {0, 1})
        self.attrs['target_shape'] = [2, 32, 48]
        self.x_dist_tensor_spec.shape = [1, 48]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1])
        self.out_dist_tensor_spec.shape = [2, 32, 48]
        self.out_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['target_shape'],
        )
        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, 1, -1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(inferred_output_dist_attrs[0]._partial_dims(), {0, 1})

        # [1, 48], [2, 32, 48]
        # [-1, 1], [0, -1, 1] --> [-1, 1], [0, -1, 1], [-1, 1](partial {0})
        self.x_dist_tensor_spec.set_dims_mapping([-1, 1])
        self.out_dist_tensor_spec.set_dims_mapping([0, -1, 1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['target_shape'],
        )
        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1, 1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, 1])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(inferred_output_dist_attrs[0]._partial_dims(), {0})

        # [1, 48], [2, 32, 48]
        # [-1, -1], [0, -1, 1] --> [-1, 1], [0, -1, 1], [-1, 1](partial {0})
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1])
        self.out_dist_tensor_spec.set_dims_mapping([0, -1, 1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['target_shape'],
        )
        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, 1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [0, -1, 1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, 1])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(inferred_output_dist_attrs[0]._partial_dims(), {0})


if __name__ == "__main__":
    unittest.main()
