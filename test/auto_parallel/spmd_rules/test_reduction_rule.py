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

from paddle.distributed.auto_parallel.static.completion import get_spmd_rule
from paddle.distributed.auto_parallel.static.dist_attribute import (
    DistTensorSpec,
    TensorDistAttr,
)
from paddle.distributed.fleet import auto


class TestReductionSPMDRule(unittest.TestCase):
    """
    Unit tests for reduction spmd rule.
    """

    def setUp(self):
        self.rule = get_spmd_rule("max")

        x_shape = [64, 32]
        process_mesh = auto.ProcessMesh(mesh=[0, 1, 2, 3])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [1, 0]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        self.attrs = {
            'keep_dim': False,
            'axis': [0],
            'linearity': False,
        }

    def test_single_mesh_dim(self):
        # reduce on dim 0, keep_dim = false
        # [0, -1] --> [0, -1], [-1], partial_on_dim:[0]
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [0]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), [0])

        # reduce on dim 0, keep_dim = true
        # [0, -1] --> [0, -1], [-1, -1], partial_on_dim:[0]
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [0]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), [0])

        # reduce on dim 1, keep_dim = false
        # [0, -1] --> [0, -1], [0], partial_on_dim:[]
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec], self.attrs
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
            [self.x_dist_tensor_spec], self.attrs
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
            [self.x_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), [0])

        # reduce on dim 0 and 1, keep_dim = true
        # [0, -1] --> [0, -1], [-1, -1], partial_on_dim:[0]
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [0, 1]
        self.x_dist_tensor_spec.set_dims_mapping([0, -1])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), [0])

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
            [self.x_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0])

        # reduce on dim 1, 2, keep_dim = false
        # [-1, 0, 1] --> [-1, 0, 1], [-1], partial_on_dim:[0, 1]
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1, 2]
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0, 1])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, 0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1])

        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), [0, 1])
        infered_output_dist_attrs[0]._clean_partial_status()
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)
        # reduction on dim 1, 2, keep_dim = false
        # [1, -1, -1] --> [1, -1, -1], [1], partial_on_dim:[]
        self.attrs['keep_dim'] = False
        self.attrs['axis'] = [1, 2]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec], self.attrs
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
            [self.x_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), [1])
        infered_output_dist_attrs[0]._clean_partial_status()
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)

        # reduction on dim 1, 2, keep_dim = true
        # [0, 1, -1] --> [0, 1, -1], [0, -1, -1], partial_on_dim:[1]
        self.attrs['keep_dim'] = True
        self.attrs['axis'] = [1, 2]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [0, 1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [0, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), [1])


if __name__ == "__main__":
    unittest.main()
