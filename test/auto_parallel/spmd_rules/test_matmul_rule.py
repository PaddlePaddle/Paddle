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


class TestMatmulSPMDRule(unittest.TestCase):
    def setUp(self):
        self.rule = get_spmd_rule("matmul")

        x_shape = [64, 32]
        y_shape = [32, 48]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [1, 0]
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        y_tensor_dist_attr = TensorDistAttr()
        y_tensor_dist_attr.dims_mapping = [0, -1]
        y_tensor_dist_attr.process_mesh = process_mesh
        self.y_dist_tensor_spec = DistTensorSpec(y_shape, y_tensor_dist_attr)

        self.attrs = {
            'trans_x': False,
            'trans_y': False,
        }

    def test_matmul_infer_forward(self):
        # TODO test partial: mk[1, 0],kn[0, -1] --> mk[1, 0],kn[0, -1] = nm[1, -1] partial[0]
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec, self.y_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(infered_input_dist_attrs), 2)
        self.assertEqual(len(infered_output_dist_attrs), 1)

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), [0])

        # test row parallel: mk[1, -1],kn[-1, -1] --> mk[1, -1],kn[-1, -1] = nm[1, -1] partial[]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec, self.y_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)

        # test row parallel: mk[1, -1],kn[-1, -1] --> mk[1, -1],kn[-1, -1] = nm[1, -1] partial[]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec, self.y_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)

        # test n parallel: mk[-1, -1],kn[-1, 0] --> mk[-1, -1],kn[-1, 0] = nm[-1, 0] partial[]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, 0])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec, self.y_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)

        # test partial with propogation: mk[1, 0],kn[-1,-1] --> mk[1, 0],kn[0, -1] = nm[1, -1] partial[0]
        self.x_dist_tensor_spec.set_dims_mapping([1, 0])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec, self.y_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [1, 0])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), [0])

        # mk[-1,-1],kn[1,0] --> mk[-1, 1],kn[1, 0] = nm[-1, 0] partial[1]:
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([1, 0])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec, self.y_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(infered_input_dist_attrs[0].dims_mapping, [-1, 1])
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [1, 0])
        self.assertEqual(infered_output_dist_attrs[0].dims_mapping, [-1, 0])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), [1])

        # abcmk[1, 0, -1, -1],kn[-1, -1] --> abcmk[1, 0, -1, -1],kn[-1, -1] = abcmn[1, 0, -1, -1] partial[]: done
        self.x_dist_tensor_spec.shape = [512, 48, 64, 32]
        self.x_dist_tensor_spec.set_dims_mapping([0, 1, -1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec, self.y_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [0, 1, -1, -1]
        )
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)

        # abcmk[1, -1, -1, 0],kn[-1, -1] --> abcmk[1, -1, -1, 0],kn[0, -1] = abcmn[1,-1, -1, -1] partial[0]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec, self.y_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0]
        )
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [0, -1])
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [1, -1, -1, -1]
        )
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), [0])

        # trans_x = True, abcmk[1, -1, -1, 0], kn[-1, -1] --> abcmk[1, -1, -1, 0],kn[-1, -1] = abcmn[1, -1, 0, -1] partial[]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])
        self.attrs['trans_x'] = True
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec, self.y_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0]
        )
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [1, -1, 0, -1]
        )
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)

        # trans_y = True, abcmk[-1, -1, -1, -1], kn[1, 0] --> abcmk[-1, -1, -1, 0],kn[1, 0] = abcmn[-1, -1, -1, 1] partial[0]: done
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, -1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([1, 0])
        self.attrs['trans_x'] = False
        self.attrs['trans_y'] = True
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec, self.y_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [-1, -1, -1, 0]
        )
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [1, 0])
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, -1, 1]
        )
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), [0])
        infered_output_dist_attrs[0]._clean_partial_dims([0])
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)

        # trans_y = True, trans_x = True, abcmk[-1, -1, 0, 1], kn[1, 0] --> abcmk[-1, -1, 0, 1]],kn[-1, 0] = abcmn[-1, -1, 1, -1] partial[0]
        # multiple mesh dim shard same tensor axis
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0, 1])
        self.y_dist_tensor_spec.set_dims_mapping([1, 0])
        self.attrs['trans_x'] = True
        self.attrs['trans_y'] = True
        result_dist_attrs = self.rule.infer_forward(
            [self.x_dist_tensor_spec, self.y_dist_tensor_spec], self.attrs
        )
        infered_input_dist_attrs = result_dist_attrs[0]
        infered_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(
            infered_input_dist_attrs[0].dims_mapping, [-1, -1, 0, 1]
        )
        self.assertEqual(infered_input_dist_attrs[1].dims_mapping, [-1, 0])
        self.assertEqual(
            infered_output_dist_attrs[0].dims_mapping, [-1, -1, 1, -1]
        )
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), True)
        self.assertEqual(infered_output_dist_attrs[0]._partial_dims(), [0])
        infered_output_dist_attrs[0]._clean_partial_status()
        self.assertEqual(infered_output_dist_attrs[0]._is_partial(), False)

        # trans_y = True, trans_x = True, abcmk[-1, -1, 1, 0], kn[1, 0] --> error:
        # one mesh dim shard multiple tensor axes
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 1, 0])
        self.y_dist_tensor_spec.set_dims_mapping([1, 0])
        self.attrs['trans_x'] = True
        self.attrs['trans_y'] = True
        with self.assertRaises(NotImplementedError):
            self.rule.infer_forward(
                [self.x_dist_tensor_spec, self.y_dist_tensor_spec], self.attrs
            )


if __name__ == "__main__":
    unittest.main()
