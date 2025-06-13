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


class TestMatmulSPMDRule(unittest.TestCase):
    def setUp(self):
        # After replaced all spmd rules by phi impl, we can recover the
        # api name to `get_spmd_rule`
        self.rule = core.get_phi_spmd_rule("fused_gemm_epilogue")

        self.attrs = OrderedDict([('trans_x', False), ('trans_y', False)])

    def test_fused_gemm_epilogue_infer_forward(self):
        x_shape = [64, 32]
        y_shape = [32, 48]
        bias_shape = [48]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        y_tensor_dist_attr = TensorDistAttr()
        y_tensor_dist_attr.process_mesh = process_mesh
        self.y_dist_tensor_spec = DistTensorSpec(y_shape, y_tensor_dist_attr)

        bias_tensor_dist_attr = TensorDistAttr()
        bias_tensor_dist_attr.process_mesh = process_mesh
        self.bias_dist_tensor_spec = DistTensorSpec(
            bias_shape, bias_tensor_dist_attr
        )
        self.bias_dist_tensor_spec.set_dims_mapping([-1])

        # has partial,force to replicate test partial: mk[1, 0],kn[0, -1],bias[-1] --> mk[1, 0],kn[0, -1],bias[-1] = nm[1, -1]
        self.x_dist_tensor_spec.set_dims_mapping([1, 0])
        self.y_dist_tensor_spec.set_dims_mapping([0, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.bias_dist_tensor_spec,
            False,
            False,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(len(result_dist_attrs), 2)
        self.assertEqual(len(inferred_input_dist_attrs), 3)
        self.assertEqual(len(inferred_output_dist_attrs), 2)

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # test row parallel: mk[1, -1],kn[-1, -1],bias[-1] --> mk[1, -1],kn[-1, -1],bias[-1] = nm[1, -1]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.bias_dist_tensor_spec,
            False,
            False,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [1, -1])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # test n parallel: mk[-1, -1],kn[-1, 0],bias[-1] --> mk[-1, -1],kn[-1, 0],bias[0] = nm[-1, 0]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, 0])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.bias_dist_tensor_spec,
            False,
            False,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, 0])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [0])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, 0])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # has partial,force to replicate test partial with propagation: mk[1, 0],kn[-1,-1],bias[-1] --> mk[1, 0],kn[0, -1],bias[-1] = nm[1, -1]
        self.x_dist_tensor_spec.set_dims_mapping([1, 0])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])
        self.bias_dist_tensor_spec.set_dims_mapping([-1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.bias_dist_tensor_spec,
            False,
            False,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # has partial,force to replicate mk[-1,-1],kn[1,0],bias[-1] --> mk[-1, 1],kn[1, 0],bias[0] = nm[-1, 0]
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([1, 0])
        self.bias_dist_tensor_spec.set_dims_mapping([-1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.bias_dist_tensor_spec,
            False,
            False,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]

        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # abcmk[1, 0, -1, -1],kn[-1, -1],bias[-1] --> abcmk[1, 0, -1, -1],kn[-1, -1],bias[-1] = abcmn[1, 0, -1, -1]
        self.x_dist_tensor_spec.shape = [512, 48, 64, 32]
        self.x_dist_tensor_spec.set_dims_mapping([1, 0, -1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.bias_dist_tensor_spec,
            False,
            False,
        )
        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [1, 0, -1, -1]
        )
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, 0, -1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # has partial,force to replicate abcmk[1, -1, -1, 0],kn[-1, -1],bias[-1] --> abcmk[1, -1, -1, 0],kn[0, -1],bias[-1] = abcmn[1,-1, -1, -1]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.bias_dist_tensor_spec,
            False,
            False,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [1, -1, -1, -1]
        )
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, -1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # trans_x = True, abcmk[1, -1, -1, 0], kn[-1, -1],bias[-1] --> abcmk[1, -1, -1, 0],kn[-1, -1],bias[-1] = abcmn[1, -1, 0, -1]
        self.x_dist_tensor_spec.set_dims_mapping([1, -1, -1, 0])
        self.y_dist_tensor_spec.set_dims_mapping([-1, -1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.bias_dist_tensor_spec,
            True,
            False,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [1, -1, -1, 0]
        )
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [1, -1, 0, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # has partial,force to replicate trans_y = True, abcmk[-1, -1, -1, -1], kn[1, 0],bias[-1] --> abcmk[-1, -1, -1, 0],kn[1, 0],bias[1] = abcmn[-1, -1, -1, 1] partial[0]: done
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, -1, -1])
        self.y_dist_tensor_spec.set_dims_mapping([1, 0])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.bias_dist_tensor_spec,
            False,
            True,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # has partial,force to replicate trans_y = True, trans_x = True, abcmk[-1, -1, 0, 1], kn[1, 0],bias[-1] --> abcmk[-1, -1, 0, 1]],kn[-1, 0],bias[-1] = abcmn[-1, -1, 1, -1]
        # multiple mesh dim shard same tensor axis
        self.x_dist_tensor_spec.set_dims_mapping([-1, -1, 0, 1])
        self.y_dist_tensor_spec.set_dims_mapping([1, 0])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.bias_dist_tensor_spec,
            True,
            True,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(
            inferred_input_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(
            inferred_output_dist_attrs[0].dims_mapping, [-1, -1, -1, -1]
        )
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # has partial,force to replicate mk[-1,1],k[1],bias[1] --> mk[-1,-1],k[-1],bias[-1] = m[-1]
        x_shape = [64, 32]
        y_shape = [32]
        bias_shape = [32]

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        y_tensor_dist_attr = TensorDistAttr()
        y_tensor_dist_attr.process_mesh = process_mesh
        self.y_dist_tensor_spec = DistTensorSpec(y_shape, y_tensor_dist_attr)

        bias_tensor_dist_attr = TensorDistAttr()
        bias_tensor_dist_attr.process_mesh = process_mesh
        self.bias_dist_tensor_spec = DistTensorSpec(
            bias_shape, bias_tensor_dist_attr
        )
        self.x_dist_tensor_spec.set_dims_mapping([-1, 1])
        self.y_dist_tensor_spec.set_dims_mapping([1])
        self.bias_dist_tensor_spec.set_dims_mapping([1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.bias_dist_tensor_spec,
            False,
            False,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1, -1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [-1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [-1])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # has partial,force to replicate k[-1],kn[-1,1],bias[-1] --> k[-1],kn[-1,1],bias[1] = n[1]
        x_shape = [32]
        y_shape = [32, 64]
        bias_shape = [64]

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        y_tensor_dist_attr = TensorDistAttr()
        y_tensor_dist_attr.process_mesh = process_mesh
        self.y_dist_tensor_spec = DistTensorSpec(y_shape, y_tensor_dist_attr)

        bias_tensor_dist_attr = TensorDistAttr()
        bias_tensor_dist_attr.process_mesh = process_mesh
        self.bias_dist_tensor_spec = DistTensorSpec(
            bias_shape, bias_tensor_dist_attr
        )
        self.x_dist_tensor_spec.set_dims_mapping([-1])
        self.y_dist_tensor_spec.set_dims_mapping([-1, 1])
        self.bias_dist_tensor_spec.set_dims_mapping([-1])

        result_dist_attrs = self.rule.infer_forward(
            self.x_dist_tensor_spec,
            self.y_dist_tensor_spec,
            self.bias_dist_tensor_spec,
            False,
            False,
        )

        inferred_input_dist_attrs = result_dist_attrs[0]
        inferred_output_dist_attrs = result_dist_attrs[1]
        self.assertEqual(inferred_input_dist_attrs[0].dims_mapping, [-1])
        self.assertEqual(inferred_input_dist_attrs[1].dims_mapping, [-1, 1])
        self.assertEqual(inferred_input_dist_attrs[2].dims_mapping, [1])
        self.assertEqual(inferred_output_dist_attrs[0].dims_mapping, [1])
        self.assertEqual(inferred_output_dist_attrs[0]._is_partial(), False)

        # k[-1],kn[1],bias[-1] --> error
        x_shape = [32]
        y_shape = [32]
        bias_shape = [1]

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.process_mesh = process_mesh
        self.x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        y_tensor_dist_attr = TensorDistAttr()
        y_tensor_dist_attr.process_mesh = process_mesh
        self.y_dist_tensor_spec = DistTensorSpec(y_shape, y_tensor_dist_attr)

        bias_tensor_dist_attr = TensorDistAttr()
        bias_tensor_dist_attr.process_mesh = process_mesh
        self.bias_dist_tensor_spec = DistTensorSpec(
            bias_shape, bias_tensor_dist_attr
        )

        self.x_dist_tensor_spec.set_dims_mapping([-1])
        self.y_dist_tensor_spec.set_dims_mapping([1])
        self.bias_dist_tensor_spec.set_dims_mapping([-1])

        with self.assertRaises(ValueError):
            result_dist_attrs = self.rule.infer_forward(
                self.x_dist_tensor_spec,
                self.y_dist_tensor_spec,
                self.bias_dist_tensor_spec,
                False,
                False,
            )


if __name__ == "__main__":
    unittest.main()
