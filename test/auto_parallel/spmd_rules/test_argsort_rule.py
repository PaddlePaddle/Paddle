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


class TestArgSortSPMDRule(unittest.TestCase):
    """
    Unit tests for argsort spmd rule.
    """

    def setUp(self):
        x_shape = [64, 32, 48]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [-1, -1, -1]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        indices_shape = list(x_shape)
        indices_tensor_dist_attr = TensorDistAttr()
        indices_tensor_dist_attr.dims_mapping = [-1, -1, -1]
        indices_tensor_dist_attr.process_mesh = process_mesh
        self.indices_dist_tensor_spec = DistTensorSpec(
            indices_shape, indices_tensor_dist_attr
        )

        out_grad_shape = list(x_shape)
        out_grad_tensor_dist_attr = TensorDistAttr()
        out_grad_tensor_dist_attr.dims_mapping = [-1, -1, -1]
        out_grad_tensor_dist_attr.process_mesh = process_mesh
        self.out_grad_dist_tensor_spec = DistTensorSpec(
            out_grad_shape, out_grad_tensor_dist_attr
        )

        self.rule = core.get_phi_spmd_rule("argsort")
        self.attrs = OrderedDict(
            {
                'axis': -1,
                'descending': False,
                'stable': False,
            }
        )

    def test_infer_spmd(self):
        # axis = -1
        # [0, -1, 1] --> [0, -1, -1]
        self.attrs['axis'] = -1
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, 1])
        dist_attr = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.attrs['axis'],
            self.attrs['descending'],
            self.attrs['stable'],
        )

        self.assertEqual(len(dist_attr), 2)

        x_tensor_dist_attr = dist_attr[0]
        y_tensor_dist_attr = dist_attr[1]

        self.assertEqual(len(x_tensor_dist_attr), 1)
        self.assertEqual(len(y_tensor_dist_attr), 2)

        self.assertEqual(x_tensor_dist_attr[0].dims_mapping, [0, -1, -1])
        self.assertEqual(y_tensor_dist_attr[0].dims_mapping, [0, -1, -1])
        self.assertEqual(y_tensor_dist_attr[1].dims_mapping, [0, -1, -1])

        # axis = -1
        # [0, 1, -1] --> [0, 1, -1]
        self.attrs['axis'] = -1
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        dist_attr = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.attrs['axis'],
            self.attrs['descending'],
            self.attrs['stable'],
        )

        self.assertEqual(len(dist_attr), 2)

        x_tensor_dist_attr = dist_attr[0]
        y_tensor_dist_attr = dist_attr[1]

        self.assertEqual(len(x_tensor_dist_attr), 1)
        self.assertEqual(len(y_tensor_dist_attr), 2)

        self.assertEqual(x_tensor_dist_attr[0].dims_mapping, [0, 1, -1])
        self.assertEqual(y_tensor_dist_attr[0].dims_mapping, [0, 1, -1])
        self.assertEqual(y_tensor_dist_attr[1].dims_mapping, [0, 1, -1])

        # axis = 1
        # [0, 1, -1] --> [0, -1, -1]
        self.attrs['axis'] = 1
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        dist_attr = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.attrs['axis'],
            self.attrs['descending'],
            self.attrs['stable'],
        )

        self.assertEqual(len(dist_attr), 2)

        x_tensor_dist_attr = dist_attr[0]
        y_tensor_dist_attr = dist_attr[1]

        self.assertEqual(len(x_tensor_dist_attr), 1)
        self.assertEqual(len(y_tensor_dist_attr), 2)

        self.assertEqual(x_tensor_dist_attr[0].dims_mapping, [0, -1, -1])
        self.assertEqual(y_tensor_dist_attr[0].dims_mapping, [0, -1, -1])
        self.assertEqual(y_tensor_dist_attr[1].dims_mapping, [0, -1, -1])

    def test_infer_grad_spmd(self):
        # axis = -1
        # [0, -1, 1] --> [0, -1, -1]
        self.attrs['axis'] = -1
        self.x_dist_tensor_spec.set_dims_mapping([0, -1, 1])
        self.indices_dist_tensor_spec.set_dims_mapping([0, -1, 1])
        self.out_grad_dist_tensor_spec.set_dims_mapping([0, -1, 1])
        dist_attr = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.indices_dist_tensor_spec,
            self.out_grad_dist_tensor_spec,
            self.attrs['axis'],
            self.attrs['descending'],
            self.attrs['stable'],
        )

        self.assertEqual(len(dist_attr), 2)
        self.assertEqual(len(dist_attr[0]), 3)
        self.assertEqual(len(dist_attr[1]), 1)
        self.assertEqual(dist_attr[0][0].dims_mapping, [0, -1, -1])
        self.assertEqual(dist_attr[0][1].dims_mapping, [0, -1, -1])
        self.assertEqual(dist_attr[0][2].dims_mapping, [0, -1, -1])
        self.assertEqual(dist_attr[1][0].dims_mapping, [0, -1, -1])

        # axis = 1
        # [0, 1, -1] --> [0, -1, -1]
        self.attrs['axis'] = 1
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        self.indices_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        self.out_grad_dist_tensor_spec.set_dims_mapping([0, 1, -1])
        dist_attr = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.indices_dist_tensor_spec,
            self.out_grad_dist_tensor_spec,
            self.attrs['axis'],
            self.attrs['descending'],
            self.attrs['stable'],
        )

        self.assertEqual(len(dist_attr), 2)
        self.assertEqual(len(dist_attr[0]), 3)
        self.assertEqual(len(dist_attr[1]), 1)
        self.assertEqual(dist_attr[0][0].dims_mapping, [0, -1, -1])
        self.assertEqual(dist_attr[0][1].dims_mapping, [0, -1, -1])
        self.assertEqual(dist_attr[0][2].dims_mapping, [0, -1, -1])
        self.assertEqual(dist_attr[1][0].dims_mapping, [0, -1, -1])


if __name__ == '__main__':
    unittest.main()
