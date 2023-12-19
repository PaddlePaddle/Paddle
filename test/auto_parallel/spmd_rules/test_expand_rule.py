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

import copy
import unittest
from collections import OrderedDict

from paddle.distributed.auto_parallel.static.dist_attribute import (
    DistTensorSpec,
    TensorDistAttr,
)
from paddle.distributed.fleet import auto
from paddle.framework import core


class TestExpandSPMDRule(unittest.TestCase):
    def setUp(self):
        self.rule = core.get_phi_spmd_rule("expand")

        x_shape = [8, 12, 1, 24]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [-1, -1, -1, -1]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        self.attrs = OrderedDict()

    def test_expand_infer_forward(self):
        # shape: [8, 12, 1, 24] --> [8, 12, 8, 24] (input --> output)
        # dims_mapping: [0, -1, -1, 1] --> [0, -1, -1, 1] [0, -1, -1, 1] (input --> input, output)
        expand_target_shape = [-1, -1, 8, 24]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1, 1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, expand_target_shape
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [0, -1, -1, 1]
        )
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [0, -1, -1, 1]
        )

        # shape: [8, 12, 1, 24] --> [8, 12, 10, 24] (input --> output)
        # dims_mapping: [0, -1, -1, 1] --> [0, -1, -1, 1] [0, -1, -1, 1] (input --> input, output)
        expand_target_shape = [-1, -1, 10, -1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1, 1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, expand_target_shape
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [0, -1, -1, 1]
        )
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [0, -1, -1, 1]
        )

        # shape: [8, 12, 1, 24] --> [1, 8, 12, 1, 24] (input --> output)
        # dims_mapping: [1, 0, -1, -1] --> [1, 0, -1, -1] [-1, 1, 0, -1, -1] (input --> input, output)
        expand_target_shape = [1, -1, -1, 1, -1]
        self.x_dist_tensor_spec.set_dims_mapping([1, 0, -1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, expand_target_shape
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [1, 0, -1, -1]
        )
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, 1, 0, -1, -1]
        )

        # shape: [8, 12, 1, 24] --> [4, 8, 12, 8, 24] (input --> output)
        # dims_mapping: [1, -1, -1, 0] --> [1, -1, -1, 0] [-1, 1, -1, -1, 0] (input --> input, output)
        expand_target_shape = [4, -1, 12, 8, -1]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, expand_target_shape
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0]
        )
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, 1, -1, -1, 0]
        )

        # shape: [8, 12, 1, 24] --> [4, 1, 8, 12, 1, 24] (input --> output)
        # dims_mapping: [1, -1, -1, 0] --> [1, -1, -1, 0] [-1, -1, 1, -1, -1, 0] (input --> input, output)
        expand_target_shape = [4, 1, -1, -1, 1, -1]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, expand_target_shape
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0]
        )
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, 1, -1, -1, 0]
        )

        # shape: [1, 12, 8, 4]  --> [8, 12, 8, 4] (input --> output)
        # dims_mapping: [-1, 0, -1, -1] --> [-1, 0, -1, -1] [-1, 0, -1, -1] (input --> input, output)
        shape = (1, 12, 8, 4)
        expand_target_shape = [8, -1, -1, -1]
        self.x_dist_tensor_spec.shape = shape
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0, -1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, expand_target_shape
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [-1, 0, -1, -1]
        )
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, 0, -1, -1]
        )

        # shape: [1, 12, 8, 4]  --> [8, 8, 12, 8, 4] (input --> output)
        # dims_mapping: [-1, 0, 1, -1] -> [-1, 0, 1, -1] [-1, -1, 0, 1, -1] (input --> input, output)
        shape = (1, 12, 8, 4)
        expand_target_shape = [8, 8, -1, -1, -1]
        self.x_dist_tensor_spec.shape = shape
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0, 1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, expand_target_shape
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [-1, 0, 1, -1]
        )
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, 0, 1, -1]
        )

    def test_expand_infer_backward(self):
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])

        output_tensor_dist_attr = TensorDistAttr()
        output_tensor_dist_attr.dims_mapping = [-1, -1, -1, -1]
        output_tensor_dist_attr.process_mesh = process_mesh
        input_tensor_dist_attr = copy.deepcopy(output_tensor_dist_attr)

        # shape: [8, 12, 1, 24] --> [8, 12, 8, 24] (input --> output)
        # dims_mapping: [0, -1, -1, 1] [-1, -1, 0, 1] --> [-1, -1, -1, 1] [-1, -1, -1, 1] (input, output --> input, output)
        self.x_dist_tensor_spec = DistTensorSpec(
            [8, 12, 1, 24], input_tensor_dist_attr
        )
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1, 1])
        self.output_dist_tensor_spec = DistTensorSpec(
            [8, 12, 8, 24], output_tensor_dist_attr
        )
        self.output_dist_tensor_spec.set_dims_mapping([-1, -1, 0, 1])

        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            [-1, -1, 8, -1],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1, 1]
        )
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1, 1]
        )

        # shape: [8, 12, 1, 24] --> [8, 12, 10, 24] (input --> output)
        # dims_mapping: [-1, -1, -1, 1] [0, -1, -1, 1] --> [0, -1, -1, 1] [0, -1, -1, 1] (input, output --> input, output)
        self.x_dist_tensor_spec = DistTensorSpec(
            [8, 12, 1, 24], input_tensor_dist_attr
        )
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, -1, 1])
        self.output_dist_tensor_spec = DistTensorSpec(
            [8, 12, 10, 24], output_tensor_dist_attr
        )
        self.output_dist_tensor_spec.set_dims_mapping([0, -1, -1, 1])

        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            [-1, -1, 10, -1],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [0, -1, -1, 1]
        )
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [0, -1, -1, 1]
        )

        # shape: [8, 12, 1, 24] --> [1, 8, 12, 1, 24] (input --> output)
        # dims_mapping: [-1, 1, -1, 0] [-1, 1, 0, -1, -1] --> [1, 0, -1, -1] [-1, 1, 0, -1, -1] (input, output --> input, output)
        self.x_dist_tensor_spec = DistTensorSpec(
            [8, 12, 1, 24], input_tensor_dist_attr
        )
        self.x_dist_tensor_spec.set_dims_mapping([-1, 1, -1, 0])
        self.output_dist_tensor_spec = DistTensorSpec(
            [1, 8, 12, 1, 24], output_tensor_dist_attr
        )
        self.output_dist_tensor_spec.set_dims_mapping([-1, 1, 0, -1, -1])

        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            [1, -1, -1, -1, -1],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [1, 0, -1, -1]
        )
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, 1, 0, -1, -1]
        )

        # shape: [8, 12, 1, 24] --> [1, 8, 12, 1, 24] (input --> output)
        # dims_mapping: [-1, 1, -1, 0] [-1, 1, 0, -1, -1] --> [1, 0, -1, -1] [-1, 1, 0, -1, -1] (input, output --> input, output)
        self.x_dist_tensor_spec = DistTensorSpec(
            [8, 12, 1, 24], input_tensor_dist_attr
        )
        self.x_dist_tensor_spec.set_dims_mapping([-1, 1, -1, 0])
        self.output_dist_tensor_spec = DistTensorSpec(
            [1, 8, 12, 1, 24], output_tensor_dist_attr
        )
        self.output_dist_tensor_spec.set_dims_mapping([-1, 1, 0, -1, -1])

        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            [1, -1, -1, -1, -1],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [1, 0, -1, -1]
        )
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, 1, 0, -1, -1]
        )

        # shape: [8, 12, 1, 24] --> [4, 8, 12, 8, 24] (input --> output)
        # dims_mapping: [1, -1, -1, 0] [0, -1, -1, 1, -1] --> [-1, -1, -1, -1] [-1, -1, -1, -1, -1] (input, output --> input, output)
        self.x_dist_tensor_spec = DistTensorSpec(
            [8, 12, 1, 24], input_tensor_dist_attr
        )
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        self.output_dist_tensor_spec = DistTensorSpec(
            [4, 8, 12, 8, 24], output_tensor_dist_attr
        )
        self.output_dist_tensor_spec.set_dims_mapping([0, -1, -1, 1, -1])

        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            [4, -1, -1, 8, -1],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1, -1, -1]
        )

        # shape: [8, 12, 1, 24] --> [4, 8, 12, 8, 24] (input --> output)
        # dims_mapping: [1, -1, -1, 0] [-1, 1, -1, -1, 0] --> [1, -1, -1, 0] [-1, 1, -1, -1, 0] (input, output --> input, output)
        self.x_dist_tensor_spec = DistTensorSpec(
            [8, 12, 1, 24], input_tensor_dist_attr
        )
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        self.output_dist_tensor_spec = DistTensorSpec(
            [4, 8, 12, 8, 24], output_tensor_dist_attr
        )
        self.output_dist_tensor_spec.set_dims_mapping([-1, 1, -1, -1, 0])

        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            [4, -1, -1, 8, -1],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0]
        )
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, 1, -1, -1, 0]
        )

        # shape: [8, 12, 1, 24] --> [4, 1, 8, 12, 1, 24] (input --> output)
        # dims_mapping: [1, -1, -1, 0] [-1, -1, 1, -1, -1, 0] --> [1, -1, -1, 0] [-1, -1, 1, -1, -1, 0] (input, output --> input, output)
        self.x_dist_tensor_spec = DistTensorSpec(
            [8, 12, 1, 24], input_tensor_dist_attr
        )
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        self.output_dist_tensor_spec = DistTensorSpec(
            [4, 1, 8, 12, 1, 24], output_tensor_dist_attr
        )
        self.output_dist_tensor_spec.set_dims_mapping([-1, -1, 1, -1, -1, 0])

        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            [4, 1, -1, -1, -1, -1],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0]
        )
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, 1, -1, -1, 0]
        )

        # shape: [1, 12, 8, 4] --> [8, 12, 8, 4] (input --> output)
        # dims_mapping: [-1, 0, -1, -1] [1, 0, -1, -1] --> [-1, 0, -1, -1] [-1, 0, -1, -1] (input, output --> input, output)
        self.x_dist_tensor_spec = DistTensorSpec(
            [1, 12, 8, 4], input_tensor_dist_attr
        )
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0, -1, -1])
        self.output_dist_tensor_spec = DistTensorSpec(
            [8, 12, 8, 4], output_tensor_dist_attr
        )
        self.output_dist_tensor_spec.set_dims_mapping([1, 0, -1, -1])

        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            [8, -1, -1, -1],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [-1, 0, -1, -1]
        )
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, 0, -1, -1]
        )

        # shape: [1, 12, 8, 4] --> [8, 8, 12, 8, 4] (input --> output)
        # dims_mapping: [-1, 0, 1, -1] [-1, -1, 0, 1, -1] --> [-1, 0, 1, -1] [-1, -1, 0, 1, -1] (input, output --> input, output)
        self.x_dist_tensor_spec = DistTensorSpec(
            [1, 12, 8, 4], input_tensor_dist_attr
        )
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0, 1, -1])
        self.output_dist_tensor_spec = DistTensorSpec(
            [8, 8, 12, 8, 4], output_tensor_dist_attr
        )
        self.output_dist_tensor_spec.set_dims_mapping([-1, -1, 0, 1, -1])

        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            [8, 8, -1, -1, -1],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [-1, 0, 1, -1]
        )
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, 0, 1, -1]
        )


if __name__ == "__main__":
    unittest.main()
