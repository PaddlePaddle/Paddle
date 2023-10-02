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


class TestCrossEntropyWithSoftmaxSPMDRule(unittest.TestCase):
    def setUp(self):
        self.rule1 = get_spmd_rule("cross_entropy_with_softmax")

        x_shape = [8, 1024, 50304]  # [batch_size, max_seq_len, vocab_size]
        label_shape = [8, 1024, 1]

        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2, 3], [4, 5, 6, 7]])
        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)
        label_tensor_dist_attr = TensorDistAttr()
        label_tensor_dist_attr.process_mesh = process_mesh
        self.lable_dist_tensor_spec = DistTensorSpec(
            label_shape, label_tensor_dist_attr
        )

        self.loss_spec = DistTensorSpec(self.lable_dist_tensor_spec)
        self.softmax_out_spec = DistTensorSpec(self.x_dist_tensor_spec)

        self.attrs = {
            'ignore_index': -1,
            'axis': -1,
            'numeric_stable_mode': True,
            'use_softmax': True,
            'soft_label': False,
        }

    def test_cross_entropy_with_softmax_infer_forward(self):
        # GPT DP case
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1])
        self.lable_dist_tensor_spec.set_dims_mapping([-1, 0, -1])

        result_dist_attrs = self.rule1.infer_forward(
            [self.x_dist_tensor_spec, self.lable_dist_tensor_spec], self.attrs
        )
        self.assertEqual(len(result_dist_attrs), 2)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 2)
        self.assertEqual(len(infered_output_dist_attrs), 2)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [1, 0, -1])

        self.assertEqual(
            infered_output_dist_attrs[1].dims_mapping, [1, 0, -1]
        )  # loss
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [1, 0, -1]
        )  # softmax output

        # GPT MP case, shard normalized axis
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0])
        self.lable_dist_tensor_spec.set_dims_mapping([-1, -1, -1])

        result_dist_attrs = self.rule1.infer_forward(
            [self.x_dist_tensor_spec, self.lable_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, -1, -1])

        self.assertEqual(
            infered_output_dist_attrs[1].dims_mapping, [-1, -1, -1]
        )  # loss
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, 0]
        )  # softmax output

        # GPT MP-DP case
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0])
        self.lable_dist_tensor_spec.set_dims_mapping([1, -1, -1])

        result_dist_attrs = self.rule1.infer_forward(
            [self.x_dist_tensor_spec, self.lable_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [1, -1, -1])

        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [1, -1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, 0])

        # Soft Label Error
        self.attrs['soft_label'] = True
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0])
        self.lable_dist_tensor_spec.set_dims_mapping([1, -1, -1])
        with self.assertRaises(ValueError):
            result_dist_attrs = self.rule1.infer_forward(
                [self.x_dist_tensor_spec, self.lable_dist_tensor_spec],
                self.attrs,
            )
        self.attrs['soft_label'] = False

        # Normalized axis
        self.attrs['axis'] = 1
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, 0])
        self.lable_dist_tensor_spec.set_dims_mapping([-1, -1, -1])
        result_dist_attrs = self.rule1.infer_forward(
            [self.x_dist_tensor_spec, self.lable_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [1, -1, 0])

        self.assertEqual(infered_output_dist_attrs[1].dims_mapping, [1, -1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1, 0])
        self.attrs['axis'] = -1

        # Soft Normalized axis Error
        self.attrs['axis'] = 1
        self.x_dist_tensor_spec.set_dims_mapping([-1, 0, -1])
        self.lable_dist_tensor_spec.set_dims_mapping([1, -1, -1])
        with self.assertRaises(ValueError):
            result_dist_attrs = self.rule1.infer_forward(
                [self.x_dist_tensor_spec, self.lable_dist_tensor_spec],
                self.attrs,
            )
        self.attrs['axis'] = -1

    def test_cross_entropy_with_softmax_infer_backward(self):
        # GPT DP case
        # [1, 0, -1], [1, 0, -1] (outputs) -->
        # [1, 0, -1], [1, 0, -1], (inputs)
        # [1, 0, -1], [1, 0, -1] (outputs)
        self.attrs['axis'] = -1
        self.attrs['use_softmax'] = True
        self.attrs['soft_label'] = False
        self.softmax_out_spec.set_dims_mapping([1, 0, -1])
        self.loss_spec.set_dims_mapping([1, 0, -1])

        result_dist_attrs = self.rule1.infer_backward(
            [self.x_dist_tensor_spec, self.lable_dist_tensor_spec],
            [self.softmax_out_spec, self.loss_spec],
            self.attrs,
        )
        self.assertEqual(len(result_dist_attrs), 2)
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(infered_input_dist_attrs), 2)
        self.assertEqual(len(infered_output_dist_attrs), 2)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [1, 0, -1])

        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [1, 0, -1]
        )  # softmax output
        self.assertEqual(
            infered_output_dist_attrs[1].dims_mapping, [1, 0, -1]
        )  # loss

        # GPT MP case, shard normalized axis
        # [-1, -1, 0], [-1, -1, -1] (outputs) -->
        # [-1, -1, 0], [-1, -1, -1], (inputs)
        # [-1, -1, 0], [-1, -1, -1] (outputs)
        self.attrs['axis'] = -1
        self.attrs['use_softmax'] = True
        self.attrs['soft_label'] = False
        self.softmax_out_spec.set_dims_mapping([-1, -1, 0])
        self.loss_spec.set_dims_mapping([-1, -1, -1])

        result_dist_attrs = self.rule1.infer_backward(
            [self.x_dist_tensor_spec, self.lable_dist_tensor_spec],
            [self.softmax_out_spec, self.loss_spec],
            self.attrs,
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, -1, -1])

        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, 0]
        )  # softmax output
        self.assertEqual(
            infered_output_dist_attrs[1].dims_mapping, [-1, -1, -1]
        )  # loss

        # GPT MP-DP case
        # [-1, -1, 0], [1, -1, -1] (outputs) -->
        # [1, -1, 0], [1, -1, -1], (inputs)
        # [1, -1, 0], [1, -1, -1] (outputs)
        self.attrs['axis'] = -1
        self.attrs['use_softmax'] = True
        self.attrs['soft_label'] = False
        self.softmax_out_spec.set_dims_mapping([-1, -1, 0])
        self.loss_spec.set_dims_mapping([1, -1, -1])

        result_dist_attrs = self.rule1.infer_backward(
            [self.x_dist_tensor_spec, self.lable_dist_tensor_spec],
            [self.softmax_out_spec, self.loss_spec],
            self.attrs,
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [1, -1, -1])

        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [1, -1, 0]
        )  # softmax output
        self.assertEqual(
            infered_output_dist_attrs[1].dims_mapping, [1, -1, -1]
        )  # loss

        # Soft Label, normalized axis = 1
        # [1, -1, 0], [1, -1, -1] (outputs) -->
        # [1, -1, 0], [1, -1, 0], (inputs)
        # [1, -1, 0], [1, -1, 0] (outputs)
        self.attrs['axis'] = 1
        self.attrs['use_softmax'] = True
        self.attrs['soft_label'] = True
        self.softmax_out_spec.set_dims_mapping([1, -1, 0])
        self.loss_spec.set_dims_mapping([1, -1, -1])
        result_dist_attrs = self.rule1.infer_backward(
            [self.x_dist_tensor_spec, self.lable_dist_tensor_spec],
            [self.softmax_out_spec, self.loss_spec],
            self.attrs,
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [1, -1, 0])

        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [1, -1, 0]
        )  # softmax output
        self.assertEqual(
            infered_output_dist_attrs[1].dims_mapping, [1, -1, 0]
        )  # loss


if __name__ == "__main__":
    unittest.main()
