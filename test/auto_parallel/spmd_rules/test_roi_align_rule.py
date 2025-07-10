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


class TestRoiAlignSPMDRule(unittest.TestCase):
    def setUp(self):
        x_shape = [2, 4, 16, 16]
        boxes_shape = [6, 6]
        boxes_num_shape = [2]
        out_shape = [6, 2, 3, 3]

        process_mesh = auto.ProcessMesh(mesh=[[0, 1], [2, 3]])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [0, 1, -1, -1]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        boxes_dist_attr = TensorDistAttr()
        boxes_dist_attr.dims_mapping = [-1, 1]
        boxes_dist_attr.process_mesh = process_mesh
        self.boxes_dist_tensor_spec = DistTensorSpec(
            boxes_shape, boxes_dist_attr
        )
        boxes_num_dist_attr = TensorDistAttr()
        boxes_num_dist_attr.dims_mapping = [-1]
        boxes_num_dist_attr.process_mesh = process_mesh
        self.boxes_num_dist_tensor_spec = DistTensorSpec(
            boxes_num_shape, boxes_num_dist_attr
        )

        out_grad_tensor_dist_attr = TensorDistAttr()
        out_grad_tensor_dist_attr.dims_mapping = [0, -1, -1, -1]
        out_grad_tensor_dist_attr.process_mesh = process_mesh
        self.out_grad_dist_tensor_spec = DistTensorSpec(
            out_shape, out_grad_tensor_dist_attr
        )

        self.rule = core.get_phi_spmd_rule("roi_align")
        self.attrs = OrderedDict()
        self.attrs['pooled_height'] = 3
        self.attrs['pooled_width'] = 3
        self.attrs['spatial_scale'] = 0.5
        self.attrs['sampling_ratio'] = -1
        self.attrs['aligned'] = True

    def test_roi_align_forward(self):
        # [0, 1, -1, -1], [-1, 1], [0] --> [-1, 1, -1, -1],[-1, -1],[-1],[-1,1,-1,-1]
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.boxes_dist_tensor_spec,
            self.boxes_num_dist_tensor_spec,
            self.attrs['pooled_height'],
            self.attrs['pooled_width'],
            self.attrs['spatial_scale'],
            self.attrs['sampling_ratio'],
            self.attrs['aligned'],
        )

        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 1)

        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, 1, -1, -1]
        )
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 1, -1, -1]
        )

        # [0, 1, -1, -1], [-1, 1], Fake --> [-1, 1, -1, -1],[-1, -1], Fake ,[-1,1,-1,-1]
        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.boxes_dist_tensor_spec,
            DistTensorSpec(),
            self.attrs['pooled_height'],
            self.attrs['pooled_width'],
            self.attrs['spatial_scale'],
            self.attrs['sampling_ratio'],
            self.attrs['aligned'],
        )
        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, 1, -1, -1]
        )
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[2], TensorDistAttr())
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 1, -1, -1]
        )

    def test_roi_align_backward(self):
        # [0, 1, -1, -1], [-1, 1], [0], [0, -1, -1, -1] --> [-1, 1, -1, -1],[-1,-1],[-1],[-1,1,-1,-1],[-1, 1, -1, -1]
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.boxes_dist_tensor_spec,
            self.boxes_num_dist_tensor_spec,
            self.out_grad_dist_tensor_spec,
            self.attrs['pooled_height'],
            self.attrs['pooled_width'],
            self.attrs['spatial_scale'],
            self.attrs['sampling_ratio'],
            self.attrs['aligned'],
        )

        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 4)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, 1, -1, -1]
        )
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_input_dist_attrs[3].dims_mapping, [-1, 1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 1, -1, -1]
        )

        # [0, 1, -1, -1], [-1, 1], Fake, [0, -1, -1, -1] --> [-1, 1, -1, -1],[-1,-1],Fake,[-1,1,-1,-1],[-1, 1, -1, -1]
        result_dist_attrs = self.rule.infer_backward(
            self.x_dist_tensor_spec,
            self.boxes_dist_tensor_spec,
            DistTensorSpec(),
            self.out_grad_dist_tensor_spec,
            self.attrs['pooled_height'],
            self.attrs['pooled_width'],
            self.attrs['spatial_scale'],
            self.attrs['sampling_ratio'],
            self.attrs['aligned'],
        )

        self.assertEqual(len(result_dist_attrs), 2)
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(len(inferred_input_dist_attrs), 4)
        self.assertEqual(len(inferred_output_dist_attrs), 1)
        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, 1, -1, -1]
        )
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[2], TensorDistAttr())
        self.assertEqual(
            inferred_input_dist_attrs[3].dims_mapping, [-1, 1, -1, -1]
        )
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, 1, -1, -1]
        )


if __name__ == "__main__":
    unittest.main()
