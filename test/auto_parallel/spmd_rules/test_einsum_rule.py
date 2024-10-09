# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


class TestEinsumSPMDRule(unittest.TestCase):

    def setUp(self):
        self.init_data()

    def init_data(self):
        self.equation = "ijk,ikl->ijl"
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])

        self.input_shapes = [[2, 3, 2], [2, 2, 3]]
        self.output_shape = [2, 3, 3]
        self.input_dims_mappings = [[-1, 0, 1], [-1, 1, 0]]
        self.out_grad_dims_mappings = [-1, 0, 1]

        # forward
        self.excepted_forward = [
            [[-1, -1, -1], [-1, -1, -1]],  # input_dims_mapping
            [-1, -1, -1],  # output_dims_mapping
        ]

        # backward
        self.excepted_backward = [
            [[-1, -1, -1], [-1, -1, -1]],  # input_dims_mapping
            [-1, -1, -1],  # output_grad_dims_mapping
            [[-1, -1, -1], [-1, -1, -1]],  # input_grad_dims_mapping
        ]

    def build_inputs(self):
        self.inputs = []
        for shape, dim_mapping in zip(
            self.input_shapes, self.input_dims_mappings
        ):
            tensor_dist_attr = TensorDistAttr()
            tensor_dist_attr.dims_mapping = dim_mapping
            tensor_dist_attr.process_mesh = self.process_mesh
            self.inputs.append(DistTensorSpec(shape, tensor_dist_attr))

    def build_outputs(self):
        tensor_dist_attr = TensorDistAttr()
        tensor_dist_attr.dims_mapping = self.out_grad_dims_mappings
        tensor_dist_attr.process_mesh = self.process_mesh
        self.output_grad = DistTensorSpec(self.output_shape, tensor_dist_attr)

    def run_infer_forward(self):
        rule = core.get_phi_spmd_rule("einsum")
        return rule.infer_forward(self.inputs, False, 3)

    def test_infer_forward(self):
        self.build_inputs()
        result_dist_attrs = self.run_infer_forward()
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        # inputs
        for input_dist_attr, excepted_dims_mapping in zip(
            infered_input_dist_attrs[0], self.excepted_forward[0]
        ):
            self.assertEqual(
                input_dist_attr.dims_mapping, excepted_dims_mapping
            )
        # output
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, self.excepted_forward[1]
        )

    def run_infer_backward(self):
        rule = core.get_phi_spmd_rule("einsum")
        return rule.infer_backward(self.inputs, self.output_grad, False)

    def test_infer_backward(self):
        self.build_inputs()
        self.build_outputs()
        result_dist_attrs = self.run_infer_backward()
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 2)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        # inputs
        for input_dist_attr, excepted_dims_mapping in zip(
            infered_input_dist_attrs[0], self.excepted_backward[0]
        ):
            self.assertEqual(
                input_dist_attr.dims_mapping, excepted_dims_mapping
            )
        # output_grad
        self.assertEqual(
            infered_input_dist_attrs[1].dims_mapping,
            self.excepted_backward[1],
        )
        # input_grad
        for input_grad_dist_attr, excepted_dims_mapping in zip(
            infered_output_dist_attrs[0], self.excepted_backward[2]
        ):
            self.assertEqual(
                input_grad_dist_attr.dims_mapping, excepted_dims_mapping
            )


class TestEinsumSPMDRule2(TestEinsumSPMDRule):
    def init_data(self):
        self.equation = "...,k->...k"
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])
        self.input_shapes = [[16, 16, 16], [4, 16, 16]]
        self.input_dims_mappings = [[-1, -1, -1], [-1, 1, 0]]
        self.output_shape = [16, 16, 16, 4, 16, 16]
        self.out_grad_dims_mappings = [-1, -1, -1, -1, 1, -1]

        # forward
        self.excepted_forward = [
            [[-1, -1, -1], [-1, 1, 0]],  # input_dims_mapping
            [-1, -1, -1, -1, 1, 0],  # output_dims_mapping
        ]

        # backward
        self.excepted_backward = [
            [[-1, -1, -1], [-1, -1, -1]],  # input_dims_mapping
            [-1, -1, -1, -1, -1, -1],  # output_grad_dims_mapping
            [[-1, -1, -1], [-1, -1, -1]],  # input_grad_dims_mapping
        ]

    def run_infer_forward(self):
        rule = core.get_phi_spmd_rule("einsum")
        return rule.infer_forward(self.inputs, True, 6)


class TestEinsumSPMDRule3(TestEinsumSPMDRule):
    def init_data(self):
        self.equation = "ij,k -> ijk"
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])
        self.input_shapes = [[16, 16, 16], [4, 16, 16]]
        self.input_dims_mappings = [[-1, -1, 1], [-1, 1, 0]]

        self.output_shape = [16, 16, 16, 4, 16, 16]
        self.out_grad_dims_mappings = [-1, -1, -1, -1, 1, 0]

        # forward
        self.excepted_forward = [
            [[-1, -1, -1], [-1, 1, 0]],  # input_dims_mapping
            [-1, -1, -1, -1, 1, 0],  # output_dims_mapping
        ]

        # backward
        self.excepted_backward = [
            [[-1, -1, -1], [-1, -1, -1]],  # input_dims_mapping
            [-1, -1, -1, -1, -1, -1],  # output_grad_dims_mapping
            [[-1, -1, -1], [-1, -1, -1]],  # input_grad_dims_mapping
        ]

    def run_infer_forward(self):
        rule = core.get_phi_spmd_rule("einsum")
        return rule.infer_forward(self.inputs, True, 6)


class TestEinsumSPMDRule4(TestEinsumSPMDRule):
    def init_data(self):
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])
        self.input_shapes = [[16], [4]]
        self.input_dims_mappings = [[0], [-1]]
        self.equation = "i,j -> ij"

        self.output_shape = [16, 4]
        self.out_grad_dims_mappings = [0, -1]

        # forward
        self.excepted_forward = [
            [[0], [-1]],  # input_dims_mapping
            [0, -1],  # output_dims_mapping
        ]

        # backward
        self.excepted_backward = [
            [[-1], [-1]],  # input_dims_mapping
            [-1, -1],  # output_grad_dims_mapping
            [[-1], [-1]],  # input_grad_dims_mapping
        ]

    def run_infer_forward(self):
        rule = core.get_phi_spmd_rule("einsum")
        return rule.infer_forward(self.inputs, True, 2)


if __name__ == "__main__":
    unittest.main()
