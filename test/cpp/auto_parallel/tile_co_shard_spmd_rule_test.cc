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

#include "paddle/phi/infermeta/spmd_rules/tile.h"
#include "test/cpp/auto_parallel/spmd_rule_test_util.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

struct TileTestCase {
  // input
  std::vector<int64_t> x_shape;
  std::vector<std::vector<int64_t>> x_dims_mapping;

  // repeat_times attribute
  phi::IntArray repeat_times;

  // output
  std::vector<std::vector<int64_t>> expected_x_dims_mapping;
  std::vector<std::vector<int64_t>> expected_out_dims_mapping;
};

struct TileGradTestCase {
  // input
  std::vector<int64_t> x_shape;
  std::vector<std::vector<int64_t>> x_dims_mapping;

  std::vector<int64_t> out_grad_shape;
  std::vector<std::vector<int64_t>> out_grad_dims_mapping;

  // repeat_times attribute
  phi::IntArray repeat_times;

  // output
  std::vector<std::vector<int64_t>> expected_x_dims_mapping;
  std::vector<std::vector<int64_t>> expected_out_grad_dims_mapping;

  std::vector<std::vector<int64_t>> expected_x_grad_dims_mapping;

  std::set<int64_t> partial_dims;
};

TEST(TileInferSpmd, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::string> dim_names = {"x", "y", "z"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<TileTestCase> test_cases = {
      // shape = [8, 16, 24], repeat_times = {2, 2, 1, 1}
      // [[0],[],[1,2]] -> [[],[],[1,2]], [[],[],[],[1,2]]
      {
          {8, 16, 24},
          {{0}, {}, {1, 2}},
          phi::IntArray({2, 2, 1, 1}),
          {{}, {}, {1, 2}},
          {{}, {}, {}, {1, 2}},
      },

      // shape = [8, 16, 24], repeat_times = {1, 2}
      // [[0,1],[],[2]] -> [[0,1],[],[]], [[0,1],[],[]]
      {
          {8, 16, 24},
          {{0, 1}, {}, {2}},
          phi::IntArray({1, 2}),
          {{0, 1}, {}, {}},
          {{0, 1}, {}, {}},
      },

      // shape = [8, 16, 24], repeat_times = {}
      // [[0,1],[],[2]] -> [[0,1],[],[2]], [[0,1],[],[2]]
      {
          {8, 16, 24},
          {{0, 1}, {}, {2}},
          phi::IntArray({}),
          {{0, 1}, {}, {2}},
          {{0, 1}, {}, {2}},
      },
  };

  for (const auto& tc : test_cases) {
    TensorDistAttr x_dist_attr = TensorDistAttr();
    x_dist_attr.set_process_mesh(process_mesh);
    x_dist_attr.set_dims_mapping(tc.x_dims_mapping);
    x_dist_attr.set_dynamic_dims(std::vector<bool>(tc.x_shape.size(), false));
    phi::distributed::DistMetaTensor x = phi::distributed::DistMetaTensor(
        common::make_ddim(tc.x_shape), x_dist_attr);

    // test forward
    phi::distributed::SpmdInfo forward_spmd_info =
        phi::distributed::TileInferSpmdDynamic(x, tc.repeat_times);
    EXPECT_EQ(forward_spmd_info.first.size(), static_cast<size_t>(1));
    EXPECT_EQ(forward_spmd_info.second.size(), static_cast<size_t>(1));
    check_multi_dims_mapping(forward_spmd_info.first[0],
                             tc.expected_x_dims_mapping);
    check_multi_dims_mapping(forward_spmd_info.second[0],
                             tc.expected_out_dims_mapping);
  }
}

TEST(TileGradInferSpmd, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::string> dim_names = {"x", "y", "z"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<TileGradTestCase> test_cases = {
      // x_shape = [8, 16, 24], out_grad_shape = [2, 16, 16, 24], repeat_times =
      // {2, 2, 1, 1}
      // [[0],[],[1,2]], [[],[],[],[1,2]] -> [[],[],[1,2]], [[],[],[],[1,2]],
      // [[],[],[1,2]], partial on {}
      {
          {8, 16, 24},
          {{0}, {}, {1, 2}},
          {2, 16, 16, 24},
          {{}, {}, {}, {1, 2}},
          phi::IntArray({2, 2, 1, 1}),
          {{}, {}, {1, 2}},
          {{}, {}, {}, {1, 2}},
          {{}, {}, {1, 2}},
          {},
      },
      // x_shape = [8, 16, 24], out_grad_shape = [8, 16, 48], repeat_times = {1,
      // 2}
      // [[0,1],[],[2]], [[0,1],[],[2]] -> [[0,1],[],[]], [[0,1],[],[]]],
      // [[0,1],[],[]], partial on {}
      {
          {8, 16, 24},
          {{0, 1}, {}, {2}},
          {8, 16, 48},
          {{0, 1}, {}, {2}},
          phi::IntArray({1, 2}),
          {{0, 1}, {}, {}},
          {{0, 1}, {}, {}},
          {{0, 1}, {}, {}},
          {},
      },

      // x_shape = [8, 16, 24], out_grad_shape = [8, 16, 24], repeat_times = {}
      // [[0,1],[],[2]], [[0],[1],[2]] -> [[0],[1],[2]], [[0],[1],[2]],
      // [[0],[1],[2]], partial on {}
      {
          {8, 16, 24},
          {{0, 1}, {}, {2}},
          {8, 16, 24},
          {{0}, {1}, {2}},
          phi::IntArray({}),
          {{0}, {1}, {2}},
          {{0}, {1}, {2}},
          {{0}, {1}, {2}},
          {},
      },

      // x_shape = [8, 16, 24], out_grad_shape = [8, 16, 16, 24], repeat_times =
      // {8, 2, 1, 1}
      // [[0],[],[]], [[1,2],[],[],[]] -> [[],[],[]], [[1,2],[],[],[]],
      // [[],[],[]], partial on {1,2}
      {
          {8, 16, 24},
          {{0}, {}, {}},
          {8, 16, 16, 24},
          {{1, 2}, {}, {}, {}},
          phi::IntArray({8, 2, 1, 1}),
          {{}, {}, {}},
          {{1, 2}, {}, {}, {}},
          {{}, {}, {}},
          {1, 2},
      },
  };
  for (const auto& tc : test_cases) {
    TensorDistAttr x_dist_attr = TensorDistAttr();
    x_dist_attr.set_process_mesh(process_mesh);
    x_dist_attr.set_dims_mapping(tc.x_dims_mapping);
    x_dist_attr.set_dynamic_dims(std::vector<bool>(tc.x_shape.size(), false));
    phi::distributed::DistMetaTensor x = phi::distributed::DistMetaTensor(
        common::make_ddim(tc.x_shape), x_dist_attr);
    TensorDistAttr out_grad_attr = TensorDistAttr();
    out_grad_attr.set_process_mesh(process_mesh);
    out_grad_attr.set_dims_mapping(tc.out_grad_dims_mapping);
    out_grad_attr.set_dynamic_dims(
        std::vector<bool>(tc.out_grad_shape.size(), false));
    phi::distributed::DistMetaTensor out_grad =
        phi::distributed::DistMetaTensor(common::make_ddim(tc.out_grad_shape),
                                         out_grad_attr);

    // test backward
    phi::distributed::SpmdInfo backward_spmd_info =
        phi::distributed::TileGradInferSpmdDynamic(
            x, out_grad, tc.repeat_times);
    EXPECT_EQ(backward_spmd_info.first.size(), static_cast<size_t>(2));
    EXPECT_EQ(backward_spmd_info.second.size(), static_cast<size_t>(1));
    check_multi_dims_mapping(backward_spmd_info.first[0],
                             tc.expected_x_dims_mapping);
    check_multi_dims_mapping(backward_spmd_info.first[1],
                             tc.expected_out_grad_dims_mapping);
    check_multi_dims_mapping(backward_spmd_info.second[0],
                             tc.expected_x_grad_dims_mapping);
  }
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
