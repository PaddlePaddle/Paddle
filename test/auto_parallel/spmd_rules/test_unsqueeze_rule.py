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


class TestUnsqueezeSPMDRule(unittest.TestCase):
    def setUp(self):
        self.rule = core.get_phi_spmd_rule("unsqueeze")

        x_shape = [8, 16]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [-1, -1]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        self.attrs = OrderedDict()

    def test_unsqueeze_infer_forward(self):
        # shape: [8, 16] --> [1, 8, 16]
        # dims_mapping: [0, 1] --> [0, 1] [-1, 0, 1]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1])
        self.attrs['axis'] = [0]
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0, 1])

        # shape: [8, 16] --> [8, 16, 1]
        # dims_mapping: [0, 1] --> [0, 1] [0, 1, -1]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1])
        self.attrs['axis'] = [-1]
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, 1, -1])

        # shape: [8, 16] --> [8, 1, 1, 16]
        # dims_mapping: [0, 1] --> [0, 1] [0, -1, -1, 1]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1])
        self.attrs['axis'] = [1, 2]
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [0, -1, -1, 1]
        )

        # shape: [8, 16] --> [1, 1, 1, 8, 16]
        # dims_mapping: [0, 1] --> [0, 1] [-1, -1, -1, 0, 1]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1])
        self.attrs['axis'] = [0, 1, 2]
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1, 0, 1]
        )

        # shape: [8, 16] --> [1, 8, 16]
        # dims_mapping: [1, 0] --> [1, 0] [-1, 1, 0]
        self.x_dist_tensor_spec.set_dims_mapping([1, 0])
        self.attrs['axis'] = [0]
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 1, 0])

        # shape: [8, 16] --> [8, 16, 1]
        # dims_mapping: [1, 0] --> [1, 0] [1, 0, -1]
        self.x_dist_tensor_spec.set_dims_mapping([1, 0])
        self.attrs['axis'] = [-1]
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, 0, -1])

        # shape: [8, 16] --> [8, 1, 1, 16]
        # dims_mapping: [1, 0] --> [1, 0] [1, -1, -1, 0]
        self.x_dist_tensor_spec.set_dims_mapping([1, 0])
        self.attrs['axis'] = [1, 2]
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [1, -1, -1, 0]
        )

        # shape: [8, 16] --> [1, 1, 1, 8, 16]
        # dims_mapping: [1, 0] --> [1, 0] [-1, -1, -1, 1, 0]
        self.x_dist_tensor_spec.set_dims_mapping([1, 0])
        self.attrs['axis'] = [0, 1, 2]
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1, 1, 0]
        )

        # shape: [1, 8, 16] --> [1, 1, 8, 16]
        # dims_mapping: [0, 1, -1] -->  [-1, 1, -1] [-1, -1, 1, -1]
        self.x_dist_tensor_spec.shape = [1, 8, 16]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        self.attrs['axis'] = [0]
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, 1, -1])
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, 1, -1]
        )

    def test_unsqueeze_infer_backward(self):
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])

        output_tensor_dist_attr = TensorDistAttr()
        output_tensor_dist_attr.dims_mapping = [-1, -1]
        output_tensor_dist_attr.process_mesh = process_mesh
        self.output_dist_tensor_spec = DistTensorSpec(
            [8, 16], output_tensor_dist_attr
        )

        # shape: [8, 16] --> [1, 8, 16] (input --> output)
        # dims_mapping: [-1, 0, 1] --> [0, 1], [-1, 0, 1] (output --> input, output)
        self.output_dist_tensor_spec.shape = [1, 8, 16]
        self.output_dist_tensor_spec.set_dims_mapping([-1, 0, 1])
        self.attrs['axis'] = [0]
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['axis'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0, 1])

        # shape: [8, 16] --> [8, 16, 1] (input --> output)
        # dims_mapping: [0, 1, -1] --> [0, 1], [0, 1, -1] (output --> input, output)
        self.output_dist_tensor_spec.shape = [8, 16, 1]
        self.output_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        self.attrs['axis'] = [-1]
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['axis'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, 1, -1])

        # shape: [8, 16] --> [8, 1, 1, 16] (input --> output)
        # dims_mapping: [0, -1, -1, 1] --> [0, 1], [0, -1, -1, 1] (output --> input, output)
        self.output_dist_tensor_spec.shape = [8, 1, 1, 16]
        self.output_dist_tensor_spec.set_dims_mapping([0, -1, -1, 1])
        self.attrs['axis'] = [1, 2]
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['axis'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [0, -1, -1, 1]
        )

        # shape: [8, 16] --> [1, 1, 1, 8, 16] (input --> output)
        # dims_mapping: [-1, -1, -1, 0, 1] --> [0, 1], [-1, -1, -1, 0, 1] (output --> input, output)
        self.output_dist_tensor_spec.shape = [1, 1, 1, 8, 16]
        self.output_dist_tensor_spec.set_dims_mapping([-1, -1, -1, 0, 1])
        self.attrs['axis'] = [0, 1, 2]
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['axis'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1])
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1, 0, 1]
        )

        # shape: [8, 16] --> [1, 8, 16] (input --> output)
        # dims_mapping: [-1, 1, 0] --> [1, 0], [-1, 1, 0] (output --> input, output)
        self.output_dist_tensor_spec.shape = [1, 8, 16]
        self.output_dist_tensor_spec.set_dims_mapping([-1, 1, 0])
        self.attrs['axis'] = [0]
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['axis'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 1, 0])

        # shape: [8, 16] --> [8, 16, 1] (input --> output)
        # dims_mapping: [1, 0, -1] --> [1, 0], [1, 0, -1] (output --> input, output)
        self.output_dist_tensor_spec.shape = [8, 16, 1]
        self.output_dist_tensor_spec.set_dims_mapping([1, 0, -1])
        self.attrs['axis'] = [-1]
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['axis'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, 0, -1])

        # shape: [8, 16] --> [8, 1, 1, 16] (input --> output)
        # dims_mapping: [1, -1, -1, 0] --> [1, 0], [1, -1, -1, 0] (output --> input, output)
        self.output_dist_tensor_spec.shape = [8, 1, 1, 16]
        self.output_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        self.attrs['axis'] = [1, 2]
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['axis'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [1, -1, -1, 0]
        )

        # shape: [8, 16] --> [1, 1, 1, 8, 16] (input --> output)
        # dims_mapping: [-1, -1, -1, 1, 0] --> [1, 0], [-1, -1, -1, 1, 0] (output --> input, output)
        self.output_dist_tensor_spec.shape = [1, 1, 1, 8, 16]
        self.output_dist_tensor_spec.set_dims_mapping([-1, -1, -1, 1, 0])
        self.attrs['axis'] = [0, 1, 2]
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['axis'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1, 1, 0]
        )

        # shape: [1, 8, 16] --> [1, 1, 8, 16] (input --> output)
        # dims_mapping: [-1, 0, 1, -1] --> [-1, 1, -1], [-1, -1, 1, -1] (output --> input, output)
        self.x_dist_tensor_spec.shape = [1, 8, 16]
        self.output_dist_tensor_spec.shape = [1, 1, 8, 16]
        self.output_dist_tensor_spec.set_dims_mapping([-1, 0, 1, -1])
        self.attrs['axis'] = [0]
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.attrs['axis'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, 1, -1])
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, 1, -1]
        )


if __name__ == "__main__":
    unittest.main()
