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


# case: bmm
class TestEinsumSPMDRule(unittest.TestCase):

    def setUp(self):
        self.init_data()
        self.init_parallel_setting()

    def init_data(self):
        self.equation = "ijk,ikl->ijl"
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])

        self.input_shapes = [[2, 4, 2], [2, 2, 4]]
        self.output_shape = [2, 4, 4]

    def init_parallel_setting(self):
        self.input_dims_mappings = [[0, -1, -1], [0, -1, -1]]
        self.out_grad_dims_mappings = [0, -1, -1]
        self.is_output_partial = False
        self.output_partial_dims = {}
        self.is_x_grad_partial = False
        self.x_grad_partial_dims = {}
        self.is_y_grad_partial = False
        self.y_grad_partial_dims = {}

        # forward
        self.excepted_forward = [
            [[0, -1, -1], [0, -1, -1]],  # input_dims_mapping
            [0, -1, -1],  # output_dims_mapping
        ]

        # backward
        self.excepted_backward = [
            [[0, -1, -1], [0, -1, -1]],  # input_dims_mapping
            [0, -1, -1],  # output_grad_dims_mapping
            [[0, -1, -1], [0, -1, -1]],  # input_grad_dims_mapping
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
        return rule.infer_forward(self.inputs, self.equation)

    def test_infer_forward(self):
        self.build_inputs()
        result_dist_attrs = self.run_infer_forward()
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 1)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        # inputs
        for input_dist_attr, excepted_dims_mapping in zip(
            inferred_input_dist_attrs[0], self.excepted_forward[0]
        ):
            self.assertEqual(
                input_dist_attr.dims_mapping, excepted_dims_mapping
            )
        # output
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, self.excepted_forward[1]
        )
        if self.is_output_partial:
            self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), True)
            self.assertEqual(
                inferred_output_dist_attrs[0]._partial_dims(),
                self.output_partial_dims,
            )
        else:
            self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

    def run_infer_backward(self):
        rule = core.get_phi_spmd_rule("einsum")
        # second argument for inner_cache (not used)
        return rule.infer_backward(
            self.inputs, self.inputs, self.output_grad, self.equation
        )

    def test_infer_backward(self):
        self.build_inputs()
        self.build_outputs()
        result_dist_attrs = self.run_infer_backward()
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 2)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        # inputs
        for input_dist_attr, excepted_dims_mapping in zip(
            inferred_input_dist_attrs[0], self.excepted_backward[0]
        ):
            self.assertEqual(
                input_dist_attr.dims_mapping, excepted_dims_mapping
            )
        # output_grad
        self.assertEqual(
            inferred_input_dist_attrs[1].dims_mapping,
            self.excepted_backward[1],
        )
        # input_grad
        for input_grad_dist_attr, excepted_dims_mapping in zip(
            inferred_output_dist_attrs[0], self.excepted_backward[2]
        ):
            self.assertEqual(
                input_grad_dist_attr.dims_mapping, excepted_dims_mapping
            )

        if self.is_x_grad_partial:
            self.assertEqual(
                inferred_output_dist_attrs[0][0]._is_partial(), True
            )
            self.assertEqual(
                inferred_output_dist_attrs[0][0]._partial_dims(),
                self.x_grad_partial_dims,
            )
        else:
            self.assertEqual(
                inferred_output_dist_attrs[0][0]._is_partial(), False
            )

        if self.is_y_grad_partial:
            self.assertEqual(
                inferred_output_dist_attrs[0][1]._is_partial(), True
            )
            self.assertEqual(
                inferred_output_dist_attrs[0][1]._partial_dims(),
                self.y_grad_partial_dims,
            )
        else:
            if len(inferred_output_dist_attrs[0]) == 2:
                self.assertEqual(
                    inferred_output_dist_attrs[0][1]._is_partial(), False
                )


class TestEinsumBMMSPMDRule2(TestEinsumSPMDRule):
    def init_parallel_setting(self):
        self.input_dims_mappings = [[-1, 0, -1], [-1, -1, -1]]
        self.out_grad_dims_mappings = [-1, 0, -1]
        self.is_output_partial = False
        self.output_partial_dims = {}
        self.is_x_grad_partial = False
        self.x_grad_partial_dims = {}
        self.is_y_grad_partial = True
        self.y_grad_partial_dims = {0}

        # forward
        self.excepted_forward = [
            [[-1, 0, -1], [-1, -1, -1]],  # input_dims_mapping
            [-1, 0, -1],  # output_dims_mapping
        ]

        # backward
        self.excepted_backward = [
            [[-1, 0, -1], [-1, -1, -1]],  # input_dims_mapping
            [-1, 0, -1],  # output_grad_dims_mapping
            [[-1, 0, -1], [-1, -1, -1]],  # input_grad_dims_mapping
        ]


class TestEinsumBMMSPMDRule3(TestEinsumSPMDRule):
    def init_parallel_setting(self):
        self.input_dims_mappings = [[-1, -1, 1], [-1, 1, -1]]
        self.out_grad_dims_mappings = [-1, -1, -1]
        self.is_output_partial = True
        self.output_partial_dims = {1}
        self.is_x_grad_partial = False
        self.x_grad_partial_dims = {}
        self.is_y_grad_partial = False
        self.y_grad_partial_dims = {}

        # forward
        self.excepted_forward = [
            [[-1, -1, 1], [-1, 1, -1]],  # input_dims_mapping
            [-1, -1, -1],  # output_dims_mapping
        ]

        # backward
        self.excepted_backward = [
            [[-1, -1, 1], [-1, 1, -1]],  # input_dims_mapping
            [-1, -1, -1],  # output_grad_dims_mapping
            [[-1, -1, 1], [-1, 1, -1]],  # input_grad_dims_mapping
        ]


class TestEinsumBMMSPMDRule4(TestEinsumSPMDRule):
    def init_parallel_setting(self):
        self.input_dims_mappings = [[-1, -1, -1], [-1, -1, 1]]
        self.out_grad_dims_mappings = [-1, -1, 1]
        self.is_output_partial = False
        self.output_partial_dims = {}
        self.is_x_grad_partial = True
        self.x_grad_partial_dims = {1}
        self.is_y_grad_partial = False
        self.y_grad_partial_dims = {}

        # forward
        self.excepted_forward = [
            [[-1, -1, -1], [-1, -1, 1]],  # input_dims_mapping
            [-1, -1, 1],  # output_dims_mapping
        ]

        # backward
        self.excepted_backward = [
            [[-1, -1, -1], [-1, -1, 1]],  # input_dims_mapping
            [-1, -1, 1],  # output_grad_dims_mapping
            [[-1, -1, -1], [-1, -1, 1]],  # input_grad_dims_mapping
        ]


class TestEinsumSumSPMDRule(TestEinsumSPMDRule):
    def init_data(self):
        self.equation = "ij->"
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])

        self.input_shapes = [[2, 4]]
        self.output_shape = []

    def init_parallel_setting(self):
        self.input_dims_mappings = [[-1, 0]]
        self.out_grad_dims_mappings = []
        self.is_output_partial = True
        self.output_partial_dims = {0}
        self.is_x_grad_partial = False
        self.x_grad_partial_dims = {}
        self.is_y_grad_partial = False
        self.y_grad_partial_dims = {}

        # forward
        self.excepted_forward = [
            [[-1, 0]],  # input_dims_mapping
            [],  # output_dims_mapping
        ]

        # backward
        self.excepted_backward = [
            [[-1, 0]],  # input_dims_mapping
            [],  # output_grad_dims_mapping
            [[-1, 0]],  # input_grad_dims_mapping
        ]


class TestEinsumSumSPMDRule2(TestEinsumSPMDRule):
    def init_data(self):
        self.equation = "ij->i"
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])

        self.input_shapes = [[2, 4]]
        self.output_shape = [2]

    def init_parallel_setting(self):
        self.input_dims_mappings = [[1, 0]]
        self.out_grad_dims_mappings = [1]
        self.is_output_partial = True
        self.output_partial_dims = {0}
        self.is_x_grad_partial = False
        self.x_grad_partial_dims = {}
        self.is_y_grad_partial = False
        self.y_grad_partial_dims = {}

        # forward
        self.excepted_forward = [
            [[1, 0]],  # input_dims_mapping
            [1],  # output_dims_mapping
        ]

        # backward
        self.excepted_backward = [
            [[1, 0]],  # input_dims_mapping
            [1],  # output_grad_dims_mapping
            [[1, 0]],  # input_grad_dims_mapping
        ]


class TestEinsumPermutationSPMDRule(TestEinsumSPMDRule):
    def init_data(self):
        self.equation = "ij->ji"
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])

        self.input_shapes = [[2, 4]]
        self.output_shape = [4, 2]

    def init_parallel_setting(self):
        self.input_dims_mappings = [[-1, 0]]
        self.out_grad_dims_mappings = [0, -1]
        self.is_output_partial = False
        self.output_partial_dims = {}
        self.is_x_grad_partial = False
        self.x_grad_partial_dims = {}
        self.is_y_grad_partial = False
        self.y_grad_partial_dims = {}

        # forward
        self.excepted_forward = [
            [[-1, 0]],  # input_dims_mapping
            [0, -1],  # output_dims_mapping
        ]

        # backward
        self.excepted_backward = [
            [[-1, 0]],  # input_dims_mapping
            [0, -1],  # output_grad_dims_mapping
            [[-1, 0]],  # input_grad_dims_mapping
        ]


class TestEinsumDiagonalSPMDRule(TestEinsumSPMDRule):
    def init_data(self):
        self.equation = "ij->ii"
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])

        self.input_shapes = [[2, 4]]
        self.output_shape = [2, 2]

    def init_parallel_setting(self):
        self.input_dims_mappings = [[0, 1]]
        self.out_grad_dims_mappings = [-1, -1]
        self.is_output_partial = True
        self.output_partial_dims = {1}
        self.is_x_grad_partial = False
        self.x_grad_partial_dims = {}
        self.is_y_grad_partial = False
        self.y_grad_partial_dims = {}

        # forward
        self.excepted_forward = [
            [[-1, 1]],  # input_dims_mapping
            [-1, -1],  # output_dims_mapping
        ]

        # backward
        self.excepted_backward = [
            [[-1, 1]],  # input_dims_mapping
            [-1, -1],  # output_grad_dims_mapping
            [[-1, 1]],  # input_grad_dims_mapping
        ]


class TestEinsumTraceSPMDRule(TestEinsumSPMDRule):
    def init_data(self):
        self.equation = "iji->i"
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])

        self.input_shapes = [[4, 2, 4]]
        self.output_shape = [4]

    def init_parallel_setting(self):
        self.input_dims_mappings = [[0, 1, 0]]
        self.out_grad_dims_mappings = [-1]
        self.is_output_partial = True
        self.output_partial_dims = {1}
        self.is_x_grad_partial = False
        self.x_grad_partial_dims = {}
        self.is_y_grad_partial = False
        self.y_grad_partial_dims = {}

        # forward
        self.excepted_forward = [
            [[-1, 1, -1]],  # input_dims_mapping
            [-1],  # output_dims_mapping
        ]

        # backward
        self.excepted_backward = [
            [[-1, 1, -1]],  # input_dims_mapping
            [-1],  # output_grad_dims_mapping
            [[-1, 1, -1]],  # input_grad_dims_mapping
        ]


class TestEinsumDotSPMDRule(TestEinsumSPMDRule):
    def init_data(self):
        self.equation = "ij,ij->i"
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])

        self.input_shapes = [[4, 2], [4, 2]]
        self.output_shape = [4]

    def init_parallel_setting(self):
        self.input_dims_mappings = [[0, 1], [0, 1]]
        self.out_grad_dims_mappings = [0]
        self.is_output_partial = True
        self.output_partial_dims = {1}
        self.is_x_grad_partial = False
        self.x_grad_partial_dims = {}
        self.is_y_grad_partial = False
        self.y_grad_partial_dims = {}

        # forward
        self.excepted_forward = [
            [[0, 1], [0, 1]],  # input_dims_mapping
            [0],  # output_dims_mapping
        ]

        # backward
        self.excepted_backward = [
            [[0, 1], [0, 1]],  # input_dims_mapping
            [0],  # output_grad_dims_mapping
            [[0, 1], [0, 1]],  # input_grad_dims_mapping
        ]


class TestEinsumMulPMDRule(TestEinsumSPMDRule):
    def init_data(self):
        self.equation = "ij,ij->ij"
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])

        self.input_shapes = [[4, 2], [4, 2]]
        self.output_shape = [4, 2]

    def init_parallel_setting(self):
        self.input_dims_mappings = [[0, 1], [0, 1]]
        self.out_grad_dims_mappings = [0, 1]
        self.is_output_partial = False
        self.output_partial_dims = {}
        self.is_x_grad_partial = False
        self.x_grad_partial_dims = {}
        self.is_y_grad_partial = False
        self.y_grad_partial_dims = {}

        # forward
        self.excepted_forward = [
            [[0, 1], [0, 1]],  # input_dims_mapping
            [0, 1],  # output_dims_mapping
        ]

        # backward
        self.excepted_backward = [
            [[0, 1], [0, 1]],  # input_dims_mapping
            [0, 1],  # output_grad_dims_mapping
            [[0, 1], [0, 1]],  # input_grad_dims_mapping
        ]


class TestEinsumOuterPMDRule(TestEinsumSPMDRule):
    def init_data(self):
        self.equation = "ij,kn->ijkn"
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])

        self.input_shapes = [[4, 2], [4, 2]]
        self.output_shape = [4, 2, 4, 2]

    def init_parallel_setting(self):
        self.input_dims_mappings = [[-1, 0], [1, -1]]
        self.out_grad_dims_mappings = [-1, 0, -1, -1]
        self.is_output_partial = False
        self.output_partial_dims = {}
        self.is_x_grad_partial = False
        self.x_grad_partial_dims = {}
        self.is_y_grad_partial = True
        self.y_grad_partial_dims = {0}

        # forward
        self.excepted_forward = [
            [[-1, 0], [-1, -1]],  # input_dims_mapping
            [-1, 0, -1, -1],  # output_dims_mapping
        ]

        # backward
        self.excepted_backward = [
            [[-1, 0], [-1, -1]],  # input_dims_mapping
            [-1, 0, -1, -1],  # output_grad_dims_mapping
            [[-1, 0], [-1, -1]],  # input_grad_dims_mapping
        ]


class TestEinsumOuterPMDRule2(TestEinsumOuterPMDRule):
    def init_parallel_setting(self):
        self.input_dims_mappings = [[-1, -1], [1, -1]]
        self.out_grad_dims_mappings = [-1, -1, 1, -1]
        self.is_output_partial = False
        self.output_partial_dims = {}
        self.is_x_grad_partial = True
        self.x_grad_partial_dims = {1}
        self.is_y_grad_partial = False
        self.y_grad_partial_dims = {}

        # forward
        self.excepted_forward = [
            [[-1, -1], [1, -1]],  # input_dims_mapping
            [-1, -1, 1, -1],  # output_dims_mapping
        ]

        # backward
        self.excepted_backward = [
            [[-1, -1], [1, -1]],  # input_dims_mapping
            [-1, -1, 1, -1],  # output_grad_dims_mapping
            [[-1, -1], [1, -1]],  # input_grad_dims_mapping
        ]


class TestEinsumBroadcastPMDRule(TestEinsumSPMDRule):
    def init_data(self):
        # original equation: 'n...jk, ...kl-> ...jl'
        # parsed in python API get 'nbjk, bkl-> bjl'
        self.equation = "nbjk,bkl->bjl"
        self.process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])

        self.input_shapes = [[8, 2, 4, 2], [1, 2, 4]]
        self.output_shape = [2, 4, 4]

    def init_parallel_setting(self):
        self.input_dims_mappings = [[0, -1, -1, 1], [-1, 1, -1]]
        self.out_grad_dims_mappings = [-1, -1, -1]
        self.is_output_partial = True
        self.output_partial_dims = {0, 1}
        self.is_x_grad_partial = False
        self.x_grad_partial_dims = {}
        self.is_y_grad_partial = True
        self.y_grad_partial_dims = {0}

        # forward
        self.excepted_forward = [
            [[0, -1, -1, 1], [-1, 1, -1]],  # input_dims_mapping
            [-1, -1, -1],  # output_dims_mapping
        ]

        # backward
        self.excepted_backward = [
            [[0, -1, -1, 1], [-1, 1, -1]],  # input_dims_mapping
            [-1, -1, -1],  # output_grad_dims_mapping
            [[0, -1, -1, 1], [-1, 1, -1]],  # input_grad_dims_mapping
        ]


if __name__ == "__main__":
    unittest.main()
