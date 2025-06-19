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

from paddle.distributed.auto_parallel.static.dist_attribute import (
    DistTensorSpec,
    TensorDistAttr,
)
from paddle.distributed.fleet import auto
from paddle.framework import core


class TestDepthwiseConv2dSPMDRule(unittest.TestCase):
    def setUp(self):
        self.rule = core.get_phi_spmd_rule("depthwise_conv2d")

    def test_depthwise_conv2d_nchw_infer_forward(self):
        # forward setup
        input_shape = [2, 4, 8, 8]
        self.data_format = "NCHW"
        filter_shape = [8, 1, 3, 3]
        process_mesh = auto.ProcessMesh(
            mesh=[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
        )

        input_tensor_dist_attr = TensorDistAttr()
        input_tensor_dist_attr.dims_mapping = [0, -1, -1, -1]
        input_tensor_dist_attr.process_mesh = process_mesh
        self.input_dist_tensor_spec = DistTensorSpec(
            input_shape, input_tensor_dist_attr
        )

        filter_tensor_dist_attr = TensorDistAttr()
        filter_tensor_dist_attr.dims_mapping = [-1, -1, -1, -1]
        filter_tensor_dist_attr.process_mesh = process_mesh
        self.filter_dist_tensor_spec = DistTensorSpec(
            filter_shape, filter_tensor_dist_attr
        )

        self.strides = [1, 1]
        self.paddings = [0, 0]
        self.padding_algorithm = "EXPLICIT"
        self.group = 4
        self.dilations = [1, 1]
        # case 1
        # input: NCHinWin[0, -1, -1, -1], filter: MCHkWk[-1, -1, -1, -1] ---> output: NMHoutWout[0, -1, -1, -1]
        result_dist_attrs = self.rule.infer_forward(
            self.input_dist_tensor_spec,
            self.filter_dist_tensor_spec,
            self.strides,
            self.paddings,
            self.padding_algorithm,
            self.group,
            self.dilations,
            self.data_format,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[1].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1, -1]
        )

        # case 2
        # input: NCHinWin[-1, -1, -1, -1], filter: MCHkWk[0, -1, -1, -1] ---> output: NMHoutWout[-1, 0, -1, -1]
        self.input_dist_tensor_spec.set_dims_mapping([-1, -1, -1, -1])
        self.filter_dist_tensor_spec.set_dims_mapping([0, -1, -1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.input_dist_tensor_spec,
            self.filter_dist_tensor_spec,
            self.strides,
            self.paddings,
            self.padding_algorithm,
            self.group,
            self.dilations,
            self.data_format,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[1].dims_mapping, [0, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 0, -1, -1]
        )

        # case 3
        # input: NCHinWin[0, -1, -1, -1], filter: MCHkWk[1, -1, -1, -1] ---> output: NMHoutWout[0, 1, -1, -1]
        self.input_dist_tensor_spec.set_dims_mapping([0, -1, -1, -1])
        self.filter_dist_tensor_spec.set_dims_mapping([1, -1, -1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.input_dist_tensor_spec,
            self.filter_dist_tensor_spec,
            self.strides,
            self.paddings,
            self.padding_algorithm,
            self.group,
            self.dilations,
            self.data_format,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[1].dims_mapping, [1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )

        # case 4
        # input: NCHinWin[-1, 0, -1, -1], filter: MCHkWk[-1, -1, -1, -1] ---> output: NMHoutWout[-1, -1, -1, -1]
        # Automatically reset dim "C" to -1
        self.input_dist_tensor_spec.set_dims_mapping([-1, 0, -1, -1])
        self.filter_dist_tensor_spec.set_dims_mapping([-1, -1, -1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.input_dist_tensor_spec,
            self.filter_dist_tensor_spec,
            self.strides,
            self.paddings,
            self.padding_algorithm,
            self.group,
            self.dilations,
            self.data_format,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[1].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )

        # case 5
        # input: NCHinWin[0, 2, -1, -1], filter: MCHkWk[1, -1, -1, -1] ---> output: NMHoutWout[0, 1, -1, -1]
        # Automatically reset dim "C" to -1
        self.input_dist_tensor_spec.set_dims_mapping([0, 2, -1, -1])
        self.filter_dist_tensor_spec.set_dims_mapping([1, -1, -1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.input_dist_tensor_spec,
            self.filter_dist_tensor_spec,
            self.strides,
            self.paddings,
            self.padding_algorithm,
            self.group,
            self.dilations,
            self.data_format,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[1].dims_mapping, [1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )

    def test_depthwise_conv2d_nhwc_infer_forward(self):
        # forward setup
        input_shape = [2, 8, 8, 4]
        self.data_format = "NHWC"
        filter_shape = [8, 1, 3, 3]
        process_mesh = auto.ProcessMesh(
            mesh=[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
        )

        input_tensor_dist_attr = TensorDistAttr()
        input_tensor_dist_attr.dims_mapping = [0, -1, -1, -1]
        input_tensor_dist_attr.process_mesh = process_mesh
        self.input_dist_tensor_spec = DistTensorSpec(
            input_shape, input_tensor_dist_attr
        )

        filter_tensor_dist_attr = TensorDistAttr()
        filter_tensor_dist_attr.dims_mapping = [-1, -1, -1, -1]
        filter_tensor_dist_attr.process_mesh = process_mesh
        self.filter_dist_tensor_spec = DistTensorSpec(
            filter_shape, filter_tensor_dist_attr
        )

        self.strides = [1, 1]
        self.paddings = [0, 0]
        self.padding_algorithm = "EXPLICIT"
        self.group = 4
        self.dilations = [1, 1]
        # case 1
        # input: NHinWinC[0, -1, -1, -1], filter: MCHkWk[-1, -1, -1, -1] ---> output: NMHoutWout[0, -1, -1, -1]
        result_dist_attrs = self.rule.infer_forward(
            self.input_dist_tensor_spec,
            self.filter_dist_tensor_spec,
            self.strides,
            self.paddings,
            self.padding_algorithm,
            self.group,
            self.dilations,
            self.data_format,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[1].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1, -1]
        )

        # case 2
        # input: NHinWinC[-1, -1, -1, -1], filter: MCHkWk[0, -1, -1, -1] ---> output: NMHoutWout[-1, 0, -1, -1]
        self.input_dist_tensor_spec.set_dims_mapping([-1, -1, -1, -1])
        self.filter_dist_tensor_spec.set_dims_mapping([0, -1, -1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.input_dist_tensor_spec,
            self.filter_dist_tensor_spec,
            self.strides,
            self.paddings,
            self.padding_algorithm,
            self.group,
            self.dilations,
            self.data_format,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[1].dims_mapping, [0, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 0, -1, -1]
        )

        # case 3
        # input: NHinWinC[0, -1, -1, -1], filter: MCHkWk[1, -1, -1, -1] ---> output: NMHoutWout[0, 1, -1, -1]
        self.input_dist_tensor_spec.set_dims_mapping([0, -1, -1, -1])
        self.filter_dist_tensor_spec.set_dims_mapping([1, -1, -1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.input_dist_tensor_spec,
            self.filter_dist_tensor_spec,
            self.strides,
            self.paddings,
            self.padding_algorithm,
            self.group,
            self.dilations,
            self.data_format,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[1].dims_mapping, [1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )

        # case 4
        # input: NHinWinC[-1, -1, -1, 0], filter: MCHkWk[-1, -1, -1, -1] ---> output: NMHoutWout[-1, -1, -1, -1]
        # Automatically reset dim "C" to -1
        self.input_dist_tensor_spec.set_dims_mapping([-1, -1, -1, 0])
        self.filter_dist_tensor_spec.set_dims_mapping([-1, -1, -1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.input_dist_tensor_spec,
            self.filter_dist_tensor_spec,
            self.strides,
            self.paddings,
            self.padding_algorithm,
            self.group,
            self.dilations,
            self.data_format,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[1].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )

    def test_depthwise_conv2d_infer_backward(self):
        # backward setup
        input_shape = [2, 4, 8, 8]
        self.data_format = "NCHW"
        filter_shape = [8, 1, 3, 3]
        output_shape = [2, 8, 6, 6]
        process_mesh = auto.ProcessMesh(
            mesh=[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
        )

        input_tensor_dist_attr = TensorDistAttr()
        input_tensor_dist_attr.dims_mapping = [-1, -1, -1, -1]
        input_tensor_dist_attr.process_mesh = process_mesh
        self.input_dist_tensor_spec = DistTensorSpec(
            input_shape, input_tensor_dist_attr
        )

        filter_tensor_dist_attr = TensorDistAttr()
        filter_tensor_dist_attr.dims_mapping = [-1, -1, -1, -1]
        filter_tensor_dist_attr.process_mesh = process_mesh
        self.filter_dist_tensor_spec = DistTensorSpec(
            filter_shape, filter_tensor_dist_attr
        )

        output_tensor_dist_attr = TensorDistAttr()
        output_tensor_dist_attr.dims_mapping = [0, 1, -1, -1]
        output_tensor_dist_attr.process_mesh = process_mesh
        self.output_dist_tensor_spec = DistTensorSpec(
            output_shape, output_tensor_dist_attr
        )

        self.strides = [1, 1]
        self.paddings = [0, 0]
        self.padding_algorithm = "EXPLICIT"
        self.group = 4
        self.dilations = [1, 1]

        # case 1:
        # Output: NMHoutWout[0, 1, -1, -1] ---> input: NCHinWin[0, -1, -1, -1], filter: MCHkWk[1, -1, -1, -1]
        # input_grad: NCHinWin[0, -1, -1, -1], filter_grad: MCHkWk[1, -1, -1, -1]
        result_dist_attrs = self.rule.infer_backward(
            self.input_dist_tensor_spec,
            self.filter_dist_tensor_spec,
            self.output_dist_tensor_spec,
            self.strides,
            self.paddings,
            self.padding_algorithm,
            self.group,
            self.dilations,
            self.data_format,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 2)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [0, -1, -1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[1].dims_mapping, [1, -1, -1, -1]
        )
        self.assertEqual(
            inferred_input_dist_attrs[2].dims_mapping, [0, 1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [0, -1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[1].dims_mapping, [1, -1, -1, -1]
        )


if __name__ == "__main__":
    unittest.main()
