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
    Unit tests for split spmd rule.
    """

    def setUp(self):
        x_shape = [64, 32, 48]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])
        self.rule = core.get_phi_spmd_rule("cumsum")

        x_dist_attr = TensorDistAttr()
        x_dist_attr.dims_mapping = [-1, -1, -1]
        x_dist_attr.process_mesh = process_mesh
        self.x_spec = DistTensorSpec(x_shape, x_dist_attr)

        self.attrs = OrderedDict()
        self.attrs['axis'] = 0
        self.attrs['flatten'] = False
        self.attrs['exclusive'] = False
        self.attrs['reverse'] = False

    def test_infer_spmd(self):
        # axis = 1
        # [-1, 0, 1] --> [-1, -1, 1], [-1, -1, 1]
        self.attrs['axis'] = 1
        self.x_spec.set_dims_mapping([-1, 0, 1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_spec,
            self.attrs['axis'],
            self.attrs['flatten'],
            self.attrs['exclusive'],
            self.attrs['reverse'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, 1])

        # axis = 0
        # [-1, 0, 1] --> [-1, 0, 1], [-1, 0, 1]
        self.attrs['axis'] = 0
        self.x_spec.set_dims_mapping([-1, 0, 1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_spec,
            self.attrs['axis'],
            self.attrs['flatten'],
            self.attrs['exclusive'],
            self.attrs['reverse'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, 0, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0, 1])

        # axis=-1, flatten = True
        # [-1, 0, 1] --> [-1, -1, -1], [-1]
        self.attrs['axis'] = -1
        self.attrs['flatten'] = True
        self.x_spec.set_dims_mapping([-1, 0, 1])
        result_dist_attrs = self.rule.infer_forward(
            self.x_spec,
            self.attrs['axis'],
            self.attrs['flatten'],
            self.attrs['exclusive'],
            self.attrs['reverse'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1])
        self.attrs['flatten'] = False

    def test_infer_spmd_reverse(self):
        self.out_spec = DistTensorSpec(self.x_spec)

        # axis = 1
        # [-1, -1, 1], [-1, -1, 1] --> [-1, -1, 1], [-1, -1, 1]
        self.attrs['axis'] = 1
        self.x_spec.set_dims_mapping([-1, -1, 1])
        self.out_spec.set_dims_mapping([-1, -1, 1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_spec,
            self.out_spec,
            self.attrs['axis'],
            self.attrs['flatten'],
            self.attrs['exclusive'],
            self.attrs['reverse'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, 1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, -1, 1])

        # axis = -1, flatten = True
        # [-1, 0, 1], [-1] --> [-1, -1, -1], [-1]
        self.attrs['axis'] = -1
        self.attrs['flatten'] = True
        self.x_spec.set_dims_mapping([-1, 0, 1])
        self.out_spec.shape = [64 * 32 * 48]
        self.out_spec.set_dims_mapping([-1])
        result_dist_attrs = self.rule.infer_backward(
            self.x_spec,
            self.out_spec,
            self.attrs['axis'],
            self.attrs['flatten'],
            self.attrs['exclusive'],
            self.attrs['reverse'],
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 1)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1])
        self.attrs['flatten'] = False
        self.out_spec.shape = [64, 32, 48]


if __name__ == "__main__":
    unittest.main()
