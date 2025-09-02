/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "test/cpp/auto_parallel/spmd_rule_test_util.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

struct ArgSortTestCase {
  // input
  std::vector<int64_t> x_shape;
  std::vector<std::vector<int64_t>> x_dims_mapping;

  // axis attribute
  int axis;

  // output
  std::vector<std::vector<int64_t>> expected_x_dims_mapping;
  std::vector<std::vector<int64_t>> expected_output_dims_mapping;
  std::vector<std::vector<int64_t>> expected_indices_dims_mapping;

  // unused attribute
  bool descending = true;
  bool stable = true;
};

struct ArgSortGradTestCase {
  // input
  std::vector<int64_t> input_shape;
  std::vector<std::vector<int64_t>> indices_dims_mapping;

  std::vector<std::vector<int64_t>> x_dims_mapping;

  std::vector<std::vector<int64_t>> out_grad_dims_mapping;

  // axis attribute
  int axis;

  // output
  std::vector<std::vector<int64_t>> expected_indices_dims_mapping;
  std::vector<std::vector<int64_t>> expected_x_dims_mapping;
  std::vector<std::vector<int64_t>> expected_out_grad_dims_mapping;

  std::vector<std::vector<int64_t>> expected_x_grad_dims_mapping;
  // unused attribute
  bool descending = true;
  bool stable = true;
};

TEST(ArgSortInferSpmd, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::string> dim_names = {"x", "y", "z"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<ArgSortTestCase> test_cases = {
      // shape = [16, 32, 48], axis = -1
      // [[0,1],[2],[]] -> [[],[2],[]], [[],[2],[]]
      {{16, 32, 48},
       {{0, 1}, {2}, {}},
       -1,
       {{0, 1}, {2}, {}},
       {{0, 1}, {2}, {}},
       {{0, 1}, {2}, {}}},

      // shape = [16, 32, 48], axis = 2
      // [[0],[],[1,2]] -> [[0],[],[]], [[0],[],[]]
      {{16, 32, 48},
       {{0}, {}, {1, 2}},
       2,
       {{0}, {}, {}},
       {{0}, {}, {}},
       {{0}, {}, {}}},

      // shape = [10, 32, 48, 24], axis = 1
      // [[0,1],[2],[],[]] -> [[0,1],[],[],[]], [[0,1],[],[],[]]
      {{10, 32, 48, 24},
       {{0, 1}, {2}, {}, {}},
       1,
       {{0, 1}, {}, {}, {}},
       {{0, 1}, {}, {}, {}},
       {{0, 1}, {}, {}, {}}}};

  for (const auto& tc : test_cases) {
    TensorDistAttr t_dist_attr = TensorDistAttr();
    t_dist_attr.set_process_mesh(process_mesh);
    t_dist_attr.set_dims_mapping(tc.x_dims_mapping);
    t_dist_attr.set_dynamic_dims(std::vector<bool>(tc.x_shape.size(), false));
    phi::distributed::DistMetaTensor x = phi::distributed::DistMetaTensor(
        common::make_ddim(tc.x_shape), t_dist_attr);

    // test forward
    phi::distributed::SpmdInfo forward_spmd_info =
        phi::distributed::ArgSortInferSpmd(
            x, tc.axis, tc.descending, tc.stable);
    EXPECT_EQ(forward_spmd_info.first.size(), static_cast<size_t>(1));
    EXPECT_EQ(forward_spmd_info.second.size(), static_cast<size_t>(2));
    check_multi_dims_mapping(forward_spmd_info.first[0],
                             tc.expected_x_dims_mapping);
    check_multi_dims_mapping(forward_spmd_info.second[0],
                             tc.expected_output_dims_mapping);
    check_multi_dims_mapping(forward_spmd_info.second[1],
                             tc.expected_indices_dims_mapping);
  }
}

TEST(ArgSortGradInferSpmd, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::string> dim_names = {"x", "y", "z"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<ArgSortGradTestCase> test_cases = {
      // shape = [16, 32, 48], axis = -1
      // [[0,1],[2],[]], [[0,1],[2],[]], [[0,1],[2],[]] -> [[0,1],[2],[]],
      // [[0,1],[2],[]], [[0,1],[2],[]], [[0,1],[2],[]]
      {{16, 32, 48},
       {{0, 1}, {2}, {}},
       {{0, 1}, {2}, {}},
       {{0, 1}, {2}, {}},
       -1,
       {{0, 1}, {2}, {}},
       {{0, 1}, {2}, {}},
       {{0, 1}, {2}, {}},
       {{0, 1}, {2}, {}}},
      // axis = 2
      // [[0,1],[],[2]], [[0,1],[],[2]], [[0,1],[],[2]] -> [[0,1],[],[]],
      // [[0,1],[],[]], [[0,1],[],[]], [[0,1],[],[]]
      {{16, 32, 48},
       {{0, 1}, {}, {2}},
       {{0, 1}, {}, {2}},
       {{0, 1}, {}, {2}},
       2,
       {{0, 1}, {}, {}},
       {{0, 1}, {}, {}},
       {{0, 1}, {}, {}},
       {{0, 1}, {}, {}}},
      // [10, 32, 48, 24], axis = 1
      // [[0],[1,2],[]], [[0],[1,2],[]], [[0],[1,2],[]] -> [[0],[],[]],
      // [[0],[],[]], [[0],[],[]], [[0],[],[]]
      {{10, 32, 48, 24},
       {{0}, {1, 2}, {}, {}},
       {{0}, {1, 2}, {}, {}},
       {{0}, {1, 2}, {}, {}},
       1,
       {{0}, {}, {}, {}},
       {{0}, {}, {}, {}},
       {{0}, {}, {}, {}},
       {{0}, {}, {}, {}}}};
  for (const auto& tc : test_cases) {
    TensorDistAttr indices_dist_attr = TensorDistAttr();
    indices_dist_attr.set_process_mesh(process_mesh);
    indices_dist_attr.set_dims_mapping(tc.indices_dims_mapping);
    indices_dist_attr.set_dynamic_dims(
        std::vector<bool>(tc.input_shape.size(), false));
    phi::distributed::DistMetaTensor indices = phi::distributed::DistMetaTensor(
        common::make_ddim(tc.input_shape), indices_dist_attr);
    TensorDistAttr x_dist_attr = TensorDistAttr();
    x_dist_attr.set_process_mesh(process_mesh);
    x_dist_attr.set_dims_mapping(tc.x_dims_mapping);
    x_dist_attr.set_dynamic_dims(
        std::vector<bool>(tc.input_shape.size(), false));
    phi::distributed::DistMetaTensor x = phi::distributed::DistMetaTensor(
        common::make_ddim(tc.input_shape), x_dist_attr);
    TensorDistAttr out_grad_dist_attr = TensorDistAttr();
    out_grad_dist_attr.set_process_mesh(process_mesh);
    out_grad_dist_attr.set_dims_mapping(tc.out_grad_dims_mapping);
    out_grad_dist_attr.set_dynamic_dims(
        std::vector<bool>(tc.input_shape.size(), false));
    phi::distributed::DistMetaTensor out_grad =
        phi::distributed::DistMetaTensor(common::make_ddim(tc.input_shape),
                                         out_grad_dist_attr);

    // test backward
    phi::distributed::SpmdInfo backward_spmd_info =
        phi::distributed::ArgSortGradInferSpmd(
            indices, x, out_grad, tc.axis, tc.descending, tc.stable);
    EXPECT_EQ(backward_spmd_info.first.size(), static_cast<size_t>(3));
    EXPECT_EQ(backward_spmd_info.second.size(), static_cast<size_t>(1));
    check_multi_dims_mapping(backward_spmd_info.first[0],
                             tc.expected_indices_dims_mapping);
    check_multi_dims_mapping(backward_spmd_info.first[1],
                             tc.expected_x_dims_mapping);
    check_multi_dims_mapping(backward_spmd_info.first[2],
                             tc.expected_out_grad_dims_mapping);
    check_multi_dims_mapping(backward_spmd_info.second[0],
                             tc.expected_x_grad_dims_mapping);
  }
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
