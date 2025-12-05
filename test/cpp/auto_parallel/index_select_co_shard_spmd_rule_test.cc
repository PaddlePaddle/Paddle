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

#include <set>
#include "test/cpp/auto_parallel/spmd_rule_test_util.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

struct IndexSelectTestCase {
  // input
  std::vector<int64_t> x_shape;
  std::vector<std::vector<int64_t>> x_dims_mapping;
  std::vector<int64_t> index_shape;
  std::vector<std::vector<int64_t>> index_dims_mapping;

  // axis attribute
  int axis;

  // output
  std::vector<std::vector<int64_t>> expected_x_dims_mapping;
  std::vector<std::vector<int64_t>> expected_index_dims_mapping;
  std::vector<std::vector<int64_t>> expected_out_dims_mapping;
};

struct IndexSelectGradTestCase {
  // input
  std::vector<int64_t> x_shape;
  std::vector<std::vector<int64_t>> x_dims_mapping;
  std::vector<int64_t> index_shape;
  std::vector<std::vector<int64_t>> index_dims_mapping;
  std::vector<int64_t> out_grad_shape;
  std::vector<std::vector<int64_t>> out_grad_dims_mapping;

  // axis attribute
  int axis;

  // output
  std::vector<std::vector<int64_t>> expected_x_dims_mapping;
  std::vector<std::vector<int64_t>> expected_index_dims_mapping;
  std::vector<std::vector<int64_t>> expected_out_grad_dims_mapping;

  std::vector<std::vector<int64_t>> expected_x_grad_dims_mapping;
  std::set<int64_t> partial_dims;
};

TEST(IndexSelectInferSpmd, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::string> dim_names = {"x", "y", "z"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<IndexSelectTestCase> test_cases = {
      // [8, 16, 32], [8], axis = 1
      // [[0,1],[2],[]], [[]] -> [[0,1],[],[]], [[]], [[0,1],[],[]]
      {{8, 16, 32},
       {{0, 1}, {2}, {}},
       {8},
       {{}},
       1,
       {{0, 1}, {}, {}},
       {{}},
       {{0, 1}, {}, {}}},

      // [8, 16, 32], [8], axis = 1
      // [[0,1],[2],[]], [[2]] -> [[0,1],[],[]], [[2]], [[0,1],[2],[]]
      {{8, 16, 32},
       {{0, 1}, {2}, {}},
       {8},
       {{2}},
       1,
       {{0, 1}, {}, {}},
       {{2}},
       {{0, 1}, {2}, {}}},

      // [8, 16, 32], [8], axis = 1
      // [[0,1],[2],[]], [[0]] -> [[0,1],[],[]], [[]], [[0,1],[],[]]
      {{8, 16, 32},
       {{0, 1}, {2}, {}},
       {8},
       {{0}},
       1,
       {{0, 1}, {}, {}},
       {{}},
       {{0, 1}, {}, {}}},

      // [8, 16, 32], [8], axis = 1
      // [[2],[],[]], [[0,1]] -> [[2],[],[]], [[0,1]], [[2],[0,1],[]]
      {{8, 16, 32},
       {{2}, {}, {}},
       {8},
       {{0, 1}},
       1,
       {{2}, {}, {}},
       {{0, 1}},
       {{2}, {0, 1}, {}}},

      // [8, 16, 32], [8], axis = 1
      // [[0],[],[]], [[0,1]] -> [[0],[],[]], [[1]], [[0],[1],[]]
      {{8, 16, 32},
       {{0}, {}, {}},
       {8},
       {{0, 1}},
       1,
       {{0}, {}, {}},
       {{1}},
       {{0}, {1}, {}}},
  };

  for (const auto& tc : test_cases) {
    TensorDistAttr x_dist_attr = TensorDistAttr();
    x_dist_attr.set_process_mesh(process_mesh);
    x_dist_attr.set_dims_mapping(tc.x_dims_mapping);
    x_dist_attr.set_dynamic_dims(std::vector<bool>(tc.x_shape.size(), false));
    phi::distributed::DistMetaTensor x = phi::distributed::DistMetaTensor(
        common::make_ddim(tc.x_shape), x_dist_attr);

    TensorDistAttr index_dist_attr = TensorDistAttr();
    index_dist_attr.set_process_mesh(process_mesh);
    index_dist_attr.set_dims_mapping(tc.index_dims_mapping);
    index_dist_attr.set_dynamic_dims(
        std::vector<bool>(tc.index_shape.size(), false));
    phi::distributed::DistMetaTensor index = phi::distributed::DistMetaTensor(
        common::make_ddim(tc.index_shape), index_dist_attr);

    // test forward
    phi::distributed::SpmdInfo forward_spmd_info =
        phi::distributed::IndexSelectInferSpmd(x, index, tc.axis);
    EXPECT_EQ(forward_spmd_info.first.size(), static_cast<size_t>(2));
    EXPECT_EQ(forward_spmd_info.second.size(), static_cast<size_t>(1));
    check_multi_dims_mapping(forward_spmd_info.first[0],
                             tc.expected_x_dims_mapping);
    check_multi_dims_mapping(forward_spmd_info.first[1],
                             tc.expected_index_dims_mapping);
    check_multi_dims_mapping(forward_spmd_info.second[0],
                             tc.expected_out_dims_mapping);
  }
}

TEST(IndexSelectGradInferSpmd, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::string> dim_names = {"x", "y", "z"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<IndexSelectGradTestCase> test_cases = {
      // [8, 16, 32], [8], [8, 8, 32], axis = 1
      // [[0,1],[2],[]], [[]], [[0,1], [], []] -> [[0,1],[],[]], [[]],
      // [[0,1],[],[]], [[0,1],[],[]]
      {{8, 16, 32},
       {{0, 1}, {2}, {}},
       {8},
       {{}},
       {8, 8, 32},
       {{0, 1}, {2}, {}},
       1,
       {{0, 1}, {}, {}},
       {{}},
       {{0, 1}, {}, {}},
       {{0, 1}, {}, {}},
       {}},

      // [8, 16, 32], [8], [8, 8, 32], axis = 1
      // [[0,1],[2],[]], [[2]], [[0,1],[2],[]] -> [[0,1],[],[]], [[2]],
      // [[0,1],[2],[]], [[0,1],[],[]]
      {{8, 16, 32},
       {{0, 1}, {2}, {}},
       {8},
       {{2}},
       {8, 8, 32},
       {{0, 1}, {2}, {}},
       1,
       {{0, 1}, {}, {}},
       {{2}},
       {{0, 1}, {2}, {}},
       {{0, 1}, {}, {}},
       {2}},

      // [8, 16, 32], [8], [8, 8, 32], axis = 1
      // [[0,1],[2],[]], [[0]], [[0,1],[],[]] -> [[0,1],[],[]], [[]],
      // [[0,1],[],[]], [[0,1],[],[]]
      {{8, 16, 32},
       {{0, 1}, {2}, {}},
       {8},
       {{0}},
       {8, 8, 32},
       {{0, 1}, {}, {}},
       1,
       {{0, 1}, {}, {}},
       {{}},
       {{0, 1}, {}, {}},
       {{0, 1}, {}, {}},
       {}},

      // [8, 16, 32], [8], [8, 8, 32], axis = 1
      // [[2],[],[]], [[0,1]], [[2],[0,1],[]] -> [[2],[],[]], [[0,1]],
      // [[2],[0,1],[]], [[2],[],[]]
      {{8, 16, 32},
       {{2}, {}, {}},
       {8},
       {{0, 1}},
       {8, 8, 32},
       {{2}, {0, 1}, {}},
       1,
       {{2}, {}, {}},
       {{0, 1}},
       {{2}, {0, 1}, {}},
       {{2}, {}, {}},
       {0, 1}},

      // [8, 16, 32], [8], [8, 8, 32],  axis = 1
      // [[0],[],[]], [[0,1]], [[0],[1],[]] -> [[0],[],[]], [[1]], [[0],[1],[]],
      // [[0],[],[]]
      {{8, 16, 32},
       {{0}, {}, {}},
       {8},
       {{0, 1}},
       {8, 8, 32},
       {{0}, {1}, {}},
       1,
       {{0}, {}, {}},
       {{1}},
       {{0}, {1}, {}},
       {{0}, {}, {}},
       {1}},
  };
  for (const auto& tc : test_cases) {
    TensorDistAttr x_dist_attr = TensorDistAttr();
    x_dist_attr.set_process_mesh(process_mesh);
    x_dist_attr.set_dims_mapping(tc.x_dims_mapping);
    x_dist_attr.set_dynamic_dims(std::vector<bool>(tc.x_shape.size(), false));
    phi::distributed::DistMetaTensor x = phi::distributed::DistMetaTensor(
        common::make_ddim(tc.x_shape), x_dist_attr);

    TensorDistAttr index_dist_attr = TensorDistAttr();
    index_dist_attr.set_process_mesh(process_mesh);
    index_dist_attr.set_dims_mapping(tc.index_dims_mapping);
    index_dist_attr.set_dynamic_dims(
        std::vector<bool>(tc.index_shape.size(), false));
    phi::distributed::DistMetaTensor index = phi::distributed::DistMetaTensor(
        common::make_ddim(tc.index_shape), index_dist_attr);

    TensorDistAttr out_grad_dist_attr = TensorDistAttr();
    out_grad_dist_attr.set_process_mesh(process_mesh);
    out_grad_dist_attr.set_dims_mapping(tc.out_grad_dims_mapping);
    out_grad_dist_attr.set_dynamic_dims(
        std::vector<bool>(tc.out_grad_shape.size(), false));
    phi::distributed::DistMetaTensor out_grad =
        phi::distributed::DistMetaTensor(common::make_ddim(tc.out_grad_shape),
                                         out_grad_dist_attr);

    // test backward
    phi::distributed::SpmdInfo backward_spmd_info =
        phi::distributed::IndexSelectGradInferSpmd(x, index, out_grad, tc.axis);
    EXPECT_EQ(backward_spmd_info.first.size(), static_cast<size_t>(3));
    EXPECT_EQ(backward_spmd_info.second.size(), static_cast<size_t>(1));
    check_multi_dims_mapping(backward_spmd_info.first[0],
                             tc.expected_x_dims_mapping);
    check_multi_dims_mapping(backward_spmd_info.first[1],
                             tc.expected_index_dims_mapping);
    check_multi_dims_mapping(backward_spmd_info.first[2],
                             tc.expected_out_grad_dims_mapping);
    check_multi_dims_mapping(backward_spmd_info.second[0],
                             tc.expected_x_grad_dims_mapping);
    if (!tc.partial_dims.empty()) {
      EXPECT_EQ(is_partial(backward_spmd_info.second[0]), true);
      check_partial_dims(backward_spmd_info.second[0], tc.partial_dims);
    }
  }
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
