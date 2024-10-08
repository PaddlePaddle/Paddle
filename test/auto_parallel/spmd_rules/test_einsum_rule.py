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
        self.rule = core.get_phi_spmd_rule("einsum")
        self.init_data()

    def init_data(self):
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])
        self.shapes = [[16, 16, 16], [4, 16, 16]]
        self.dim_mappings = [[-1, 0, 1], [-1, 1, 0]]
        self.equation = "mij,mlh->ljm"

        self.excepted_input_dims_mappings = [[-1, -1, -1], [-1, -1, -1]]
        self.excepted_output_dims_mapping = [-1, -1, -1]

    def build_inputs(self):
        self.inputs = []
        for shape, dim_mapping in zip(self.shapes, self.dim_mappings):
            tensor_dist_attr = TensorDistAttr()
            tensor_dist_attr.dims_mapping = dim_mapping
            tensor_dist_attr.process_mesh = self.process_mesh
            self.inputs.append(DistTensorSpec(shape, tensor_dist_attr))

    def run_spmd(self):
        return self.rule.infer_forward(self.inputs, False, 3)

    def test_infer_forward(self):
        self.build_inputs()
        result_dist_attrs = self.run_spmd()
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        x_dims_mapping = infered_input_dist_attrs[0][0].dims_mapping
        y_dims_mapping = infered_input_dist_attrs[0][1].dims_mapping
        for input_dist_attr, excepted_dims_mapping in zip(
            infered_input_dist_attrs[0], self.excepted_input_dims_mappings
        ):
            self.assertEqual(
                input_dist_attr.dims_mapping, excepted_dims_mapping
            )

        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping,
            self.excepted_output_dims_mapping,
        )


class TestEinsumSPMDRule2(TestEinsumSPMDRule):
    def init_data(self):
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])
        self.shapes = [[16, 16, 16], [4, 16, 16]]
        self.dim_mappings = [[-1, -1, -1], [-1, 1, 0]]
        self.equation = "...,k->...k"

        self.excepted_input_dims_mappings = [[-1, -1, -1], [-1, 1, 0]]
        self.excepted_output_dims_mapping = [-1, -1, -1, -1, 1, 0]

    def run_spmd(self):
        return self.rule.infer_forward(self.inputs, True)


class TestEinsumSPMDRule3(TestEinsumSPMDRule):
    def init_data(self):
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])
        self.shapes = [[16, 16, 16], [4, 16, 16]]
        self.dim_mappings = [[-1, -1, 1], [-1, 1, 0]]
        self.equation = "ij,k -> ijk"

        self.excepted_input_dims_mappings = [[-1, -1, -1], [-1, 1, 0]]
        self.excepted_output_dims_mapping = [-1, -1, -1, -1, 1, 0]

    def run_spmd(self):
        return self.rule.infer_forward(self.inputs, True)


class TestEinsumSPMDRule4(TestEinsumSPMDRule):
    def init_data(self):
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])
        self.shapes = [[16], [4]]
        self.dim_mappings = [[0], [-1]]
        self.equation = "i,j -> ij"

        self.excepted_input_dims_mappings = [[0], [-1]]
        self.excepted_output_dims_mapping = [0, -1]

    def run_spmd(self):
        return self.rule.infer_forward(self.inputs, True)


if __name__ == "__main__":
    unittest.main()
