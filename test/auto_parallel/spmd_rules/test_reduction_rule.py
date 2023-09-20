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


class TestReductionSPMDRule(unittest.TestCase):
    """
    Unit tests for reduction spmd rule.
    """

    def setUp(self):
        self.rule = core.get_phi_spmd_rule("max")

        x_shape = [64, 32]
        process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [1, 0]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        self.out_dist_tensor_spec = DistTensorSpec(self.x_dist_tensor_spec)

        self.attrs = OrderedDict([('axis', [0]), ('keep_dim', False)])

    def test_single_mesh_dim(self):
        # reduce on dim 0, keep_dim = false
        # [0, -1] --> [0, -1], [-1], partial_on_dim:[0]
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [0]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {0})

        # reduce on dim 0, keep_dim = true
        # [0, -1] --> [0, -1], [-1, -1], partial_on_dim:[0]
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [0]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {0})

        # reduce on dim 1, keep_dim = false
        # [0, -1] --> [0, -1], [0], partial_on_dim:[]
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)

        # reduce on dim 1, keep_dim = true
        # [0, -1] --> [0, -1], [0, -1], partial_on_dim:[]
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)

        # reduce on dim 0 and 1, keep_dim = false
        # [0, -1] --> [0, -1], [], partial_on_dim:[0]
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [0, 1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {0})

        # reduce on dim 0 and 1, keep_dim = true
        # [0, -1] --> [0, -1], [-1, -1], partial_on_dim:[0]
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [0, 1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {0})

    def test_multi_mesh_dim(self):
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        self.x_dist_tensor_spec.set_process_mesh(process_mesh)
        self.x_dist_tensor_spec.shape = [96, 24, 48]

        # reduce on dim 1, 2, keep_dim = false
        # [0, -1, -1] --> [0, -1, -1], [0], partial_on_dim:[]
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1, 2]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)

        # reduce on dim 1, 2, keep_dim = false
        # [-1, 0, 1] --> [-1, 0, 1], [-1], partial_on_dim:[0, 1]
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1, 2]
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0, 1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, 0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1])

        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {0, 1})
        infered_output_dist_attrs[0]._clean_partial_status()
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)
        # reduction on dim 1, 2, keep_dim = false
        # [1, -1, -1] --> [1, -1, -1], [1], partial_on_dim:[]
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1, 2]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)

        # reduction on dim 1, 2, keep_dim = false
        # [0, 1, -1] --> [0, 1, -1], [0], partial_on_dim:[1]
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1, 2]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {1})
        infered_output_dist_attrs[0]._clean_partial_status()
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)

        # reduction on dim 1, 2, keep_dim = true
        # [0, 1, -1] --> [0, 1, -1], [0, -1, -1], partial_on_dim:[1]
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [1, 2]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec, self.attrs['axis'], self.attrs['keep_dim']
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), {1})

    def test_backward_single_mesh_dim(self):
        # reduce on dim 0, keep_dim = false
        # [-1] --> [-1, -1], [-1] (output --> input, output)
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [0]
        self.out_dist_tensor_spec.shape = [32]
        self.out_dist_tensor_spec.set_dims_mapping([-1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['axis'],
            self.attrs['keep_dim'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1])

        # reduce on dim 0, keep_dim = true
        # [-1, -1] --> [-1, -1], [-1, -1] (output --> input, output)
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [0]
        self.out_dist_tensor_spec.shape = [1, 32]
        self.out_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['axis'],
            self.attrs['keep_dim'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1])

        # reduce on dim 1, keep_dim = false
        # [0] --> [0, -1], [0] (output --> input, output)
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1]
        self.out_dist_tensor_spec.shape = [64]
        self.out_dist_tensor_spec.set_dims_mapping([0])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['axis'],
            self.attrs['keep_dim'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0])

        # reduce on dim 1, keep_dim = true
        # [0, -1] --> [0, -1], [0, -1] (output --> input, output)
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [1]
        self.out_dist_tensor_spec.shape = [64, 1]
        self.out_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['axis'],
            self.attrs['keep_dim'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1])

        # reduce on dim 0 and 1, keep_dim = false
        # [] --> [-1, -1], [] (output --> input, output)
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [0, 1]
        self.out_dist_tensor_spec.shape = []
        self.out_dist_tensor_spec.set_dims_mapping([])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['axis'],
            self.attrs['keep_dim'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [])

        # reduce on dim 0 and 1, keep_dim = true
        # [-1, -1] --> [-1, -1], [-1, -1] (output --> input, output)
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [0, 1]
        self.out_dist_tensor_spec.shape = [1, 1]
        self.out_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['axis'],
            self.attrs['keep_dim'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1])

    def test_backward_multi_mesh_dim(self):
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])
        self.x_dist_tensor_spec.set_process_mesh(process_mesh)
        self.x_dist_tensor_spec.shape = [96, 24, 48]
        self.out_dist_tensor_spec.set_process_mesh(process_mesh)

        # reduce on dim 1, 2, keep_dim = false
        # [0] --> [0, -1, -1], [0] (output --> input, output)
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1, 2]
        self.out_dist_tensor_spec.shape = [96]
        self.out_dist_tensor_spec.set_dims_mapping([0])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['axis'],
            self.attrs['keep_dim'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0])

        # reduce on dim 1, 2, keep_dim = false
        # [-1] --> [-1, -1, -1], [-1] (output --> input, output)
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1, 2]
        self.out_dist_tensor_spec.shape = [96]
        self.out_dist_tensor_spec.set_dims_mapping([-1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['axis'],
            self.attrs['keep_dim'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1])

        # reduction on dim 1, 2, keep_dim = false
        # [1] --> [1, -1, -1], [1] (output --> input, output)
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1, 2]
        self.out_dist_tensor_spec.shape = [96]
        self.out_dist_tensor_spec.set_dims_mapping([1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['axis'],
            self.attrs['keep_dim'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1])

        # reduction on dim 1, 2, keep_dim = true
        # [0, -1, -1] --> [0, -1, -1], [0, -1, -1] (output --> input, output)
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [1, 2]
        self.out_dist_tensor_spec.shape = [96, 1, 1]
        self.out_dist_tensor_spec.set_dims_mapping([0, -1, -1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['axis'],
            self.attrs['keep_dim'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1, -1])

    def test_backward_multi_mesh_dim_parital(self):
        # reduction on dim 1, 2, keep_dim = true, partial_dim=[1]
        # [0, -1, -1] --> [0, -1, -1], [0, -1, -1] (output --> input, output)
        # output parital_dim: [1], input parital_dim: []
        out_shape = [96, 1, 1]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])

        self.x_dist_tensor_spec.set_process_mesh(process_mesh)
        self.x_dist_tensor_spec.shape = [96, 24, 48]
        out_tensor_dist_attr = TensorDistAttr()
        out_tensor_dist_attr.dims_mapping = [0, -1, -1]
        out_tensor_dist_attr.process_mesh = process_mesh
        out_tensor_dist_attr._set_partial_dims([1])
        self.out_dist_tensor_spec = DistTensorSpec(
            out_shape, out_tensor_dist_attr
        )

        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [1, 2]
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.out_dist_tensor_spec,
            self.attrs['axis'],
            self.attrs['keep_dim'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_input_dist_attrs[0]._is_partial(), False)


if __name__ == "__main__":
    unittest.main()
