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

from paddle.distributed.auto_parallel.completion import get_spmd_rule
from paddle.distributed.auto_parallel.dist_attribute import (
    DistTensorSpec,
    TensorDistAttr,
)
from paddle.distributed.fleet import auto


class TestMatmulSPMDRule(unittest.TestCase):
    def setUp(self):
        self.rule = get_spmd_rule("matmul")
        print("rule", self.rule, type(self.rule))

        x_shape = [64, 32]
        y_shape = [32, 48]
        process_mesh = auto.ProcessMesh(mesh=[[0, 1, 2], [3, 4, 5]])

        x_tensor_dist_attr = TensorDistAttr()
        x_tensor_dist_attr.dims_mapping = [1, 0]
        x_tensor_dist_attr.process_mesh = process_mesh
        x_dist_tensor_spec = DistTensorSpec(x_shape, x_tensor_dist_attr)

        y_tensor_dist_attr = TensorDistAttr()
        y_tensor_dist_attr.dims_mapping = [0, -1]
        y_tensor_dist_attr.process_mesh = process_mesh
        y_dist_tensor_spec = DistTensorSpec(y_shape, y_tensor_dist_attr)

        attrs = {
            'trans_x': False,
            'trans_y': False,
        }

    def test_matmul(self):
        print("rule", self.rule, type(self.rule))


#   TensorDistAttr x_dist_attr = TensorDistAttr();
#   x_dist_attr.set_process_mesh(process_mesh);
#   x_dist_attr.set_dims_mapping(std::vector<int64_t>({1, -1}));
#   x_dist_attr.set_batch_dim(-1);
#   x_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

#   TensorDistAttr y_dist_attr = TensorDistAttr();
#   y_dist_attr.set_process_mesh(process_mesh);
#   y_dist_attr.set_dims_mapping(std::vector<int64_t>({-1, -1}));
#   y_dist_attr.set_batch_dim(-1);
#   y_dist_attr.set_dynamic_dims(std::vector<bool>({false, false}));

#   DistTensorSpec x_dist_tensor_spec = DistTensorSpec(x_shape, x_dist_attr);
#   DistTensorSpec y_dist_tensor_spec = DistTensorSpec(y_shape, y_dist_attr);

#   paddle::framework::AttributeMap attrs;
#   attrs["trans_x"] = false;
#   attrs["trans_y"] = false;

#   const SPMDRuleBase& matmul_rule = SPMDRuleMap::Instance().Get(matmul);


# def test_constructor(self):
#     tensor_shape = [6, 12]
#     tensor_dist_attr = TensorDistAttr()
#     tensor_dist_attr.dims_mapping = [1, 0]
#     tensor_dist_attr.process_mesh = auto.ProcessMesh(
#         mesh=[[0, 1, 2], [3, 4, 5]]
#     )

#     spec = DistTensorSpec(tensor_shape, tensor_dist_attr)

#     self.assertEqual(spec.shape, [6, 12])
#     self.assertEqual(spec.get_dims_mapping(), [1, 0])
#     self.assertEqual(
#         spec.get_process_mesh().process_ids, [0, 1, 2, 3, 4, 5]
#     )

# def test_instance_method(self):
#     tensor_dist_attr = TensorDistAttr()
#     tensor_dist_attr.dims_mapping = [1, 0]
#     tensor_dist_attr.process_mesh = auto.ProcessMesh(
#         mesh=[[0, 1, 2], [3, 4, 5]]
#     )

#     spec = DistTensorSpec()
#     spec.set_dims_mapping(tensor_dist_attr.dims_mapping)
#     spec.set_process_mesh(tensor_dist_attr.process_mesh)

#     self.assertEqual(spec.get_dims_mapping(), [1, 0])
#     self.assertEqual(
#         spec.get_process_mesh().process_ids, [0, 1, 2, 3, 4, 5]
#     )

# def test_wrap_func(self):
#     paddle.enable_static()
#     main_program = paddle.static.Program()
#     startup_program = paddle.static.Program()

#     with paddle.static.program_guard(main_program, startup_program):
#         input = paddle.static.data(name="input", shape=[6, 12])
#         weight = paddle.static.data(name="weight", shape=[12, 4])
#         output = paddle.matmul(input, weight)

#     ops = main_program.global_block().ops
#     for idx, op in enumerate(ops):
#         if op.type == 'matmul_v2' or op.type == 'matmul':
#             op_dist_attr = OperatorDistAttr()
#             X = op.input_arg_names[0]
#             Y = op.input_arg_names[1]
#             out = op.output_arg_names[0]
#             op_dist_attr.set_input_dims_mapping(X, [-1, 1])
#             op_dist_attr.set_input_dims_mapping(Y, [1, -1])
#             op_dist_attr.set_output_dims_mapping(out, [-1, -1])
#             dist_op = DistributedOperator(op, op_dist_attr)
#             input_names = [X, Y]
#             output_names = [out]
#             attr_names = []
#             input_spec, output_spec, attrs = wrap_data_for_completion(
#                 dist_op, input_names, output_names, attr_names
#             )

#             self.assertEqual(input_spec[0].shape, [6, 12])
#             self.assertEqual(input_spec[1].shape, [12, 4])
#             self.assertEqual(output_spec[0].shape, [6, 4])
#             self.assertEqual(input_spec[0].get_dims_mapping(), [-1, 1])
#             self.assertEqual(input_spec[1].get_dims_mapping(), [1, -1])
#             self.assertEqual(output_spec[0].get_dims_mapping(), [-1, -1])


if __name__ == "__main__":
    unittest.main()
