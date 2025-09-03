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

struct SoftmaxTestCase {
  // input
  std::vector<int64_t> input_shape;
  std::vector<std::vector<int64_t>> input_dims_mapping;

  // axis attribute
  int axis;

  // output
  std::vector<std::vector<int64_t>> expected_input_dims_mapping;
  std::vector<std::vector<int64_t>> expected_output_dims_mapping;
};

struct SoftmaxGradTestCase {
  // input
  std::vector<int64_t> out_shape;
  std::vector<std::vector<int64_t>> out_dims_mapping;

  std::vector<int64_t> out_grad_shape;
  std::vector<std::vector<int64_t>> out_grad_dims_mapping;

  // axis attribute
  int axis;

  // output
  std::vector<std::vector<int64_t>> expected_out_dims_mapping;
  std::vector<std::vector<int64_t>> expected_out_grad_dims_mapping;

  std::vector<std::vector<int64_t>> expected_x_grad_dims_mapping;
};

TEST(SoftmaxInferSpmd, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::string> dim_names = {"x", "y", "z"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<SoftmaxTestCase> test_cases = {
      // shape = [32, 48, 128], axis = 0
      // [[0,1],[2],[]] -> [[],[2],[]], [[],[2],[]]
      {{32, 48, 128}, {{0, 1}, {2}, {}}, 0, {{}, {2}, {}}, {{}, {2}, {}}},
      {{32, 48, 128}, {{0, 1}, {2}, {}}, -3, {{}, {2}, {}}, {{}, {2}, {}}},

      // shape = [32, 48, 128], axis = 1
      // [[0,1],[2],[]] -> [[0, 1],[],[]], [[0, 1],[],[]]
      {{32, 48, 128},
       {{0, 1}, {2}, {}},
       1,
       {{0, 1}, {}, {}},
       {{0, 1}, {}, {}}}};

  for (const auto& tc : test_cases) {
    TensorDistAttr t_dist_attr = TensorDistAttr();
    t_dist_attr.set_process_mesh(process_mesh);
    t_dist_attr.set_dims_mapping(tc.input_dims_mapping);
    t_dist_attr.set_dynamic_dims(
        std::vector<bool>(tc.input_shape.size(), false));
    phi::distributed::DistMetaTensor x = phi::distributed::DistMetaTensor(
        common::make_ddim(tc.input_shape), t_dist_attr);

    // test forward
    phi::distributed::SpmdInfo forward_spmd_info =
        phi::distributed::SoftmaxInferSpmd(x, tc.axis);
    EXPECT_EQ(forward_spmd_info.first.size(), static_cast<size_t>(1));
    EXPECT_EQ(forward_spmd_info.second.size(), static_cast<size_t>(1));
    check_multi_dims_mapping(forward_spmd_info.first[0],
                             tc.expected_input_dims_mapping);
    check_multi_dims_mapping(forward_spmd_info.second[0],
                             tc.expected_output_dims_mapping);
  }
}

TEST(SoftmaxGradInferSpmd, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::string> dim_names = {"x", "y", "z"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<SoftmaxGradTestCase> test_cases = {
      // out_shape = [32, 48, 128], out_grad_shape = [32, 48, 128], axis = 0
      // [[0,1],[2],[]], [[0,1],[2],[]] -> [[],[2],[]], [[],[2],[]], [[],[2],[]]
      {{32, 48, 128},
       {{0, 1}, {2}, {}},
       {32, 48, 128},
       {{0, 1}, {2}, {}},
       0,
       {{}, {2}, {}},
       {{}, {2}, {}},
       {{}, {2}, {}}},
      // axis = 0
      // [[0,1],[2],[]], [[0],[1,2],[]] -> [[],[1,2],[]], [[],[1, 2],[]],
      // [[],[1,2],[]]
      {{32, 48, 128},
       {{0, 1}, {2}, {}},
       {32, 48, 128},
       {{0}, {1, 2}, {}},
       0,
       {{}, {1, 2}, {}},
       {{}, {1, 2}, {}},
       {{}, {1, 2}, {}}},
      // axis = 1
      // [[0,1],[2],[]], [[2],[0,1],[]] -> [[0,1,2],[],[]], [[0, 1, 2],[],[]],
      // [[0, 1, 2],[],[]]
      {{32, 48, 128},
       {{0, 1}, {2}, {}},
       {32, 48, 128},
       {{2}, {0, 1}, {}},
       1,
       {{0, 1, 2}, {}, {}},
       {{0, 1, 2}, {}, {}},
       {{0, 1, 2}, {}, {}}},
      // axis = 2
      // [[0],[1],[]], [[],[0,1],[]] -> [[],[0,1],[]], [[],[0,1],[]],
      // [[],[0,1],[]]
      {{32, 48, 128},
       {{0}, {1}, {}},
       {32, 48, 128},
       {{}, {0, 1}, {}},
       2,
       {{}, {0, 1}, {}},
       {{}, {0, 1}, {}},
       {{}, {0, 1}, {}}},
      // axis = 2
      // [[0],[1],[]], [[0,1],[],[]] -> [[0,1],[],[]], [[0, 1],[],[]],
      // [[0,1],[],[]]
      {{32, 48, 128},
       {{0}, {1}, {}},
       {32, 48, 128},
       {{0, 1}, {}, {}},
       2,
       {{0, 1}, {}, {}},
       {{0, 1}, {}, {}},
       {{0, 1}, {}, {}}},
      // axis = 2
      // [[0],[1,2],[]], [[],[0,1],[]] -> [[0],[1,2],[]], [[0],[1,2],[]],
      // [[0],[1,2],[]]
      {{32, 48, 128},
       {{0}, {1, 2}, {}},
       {32, 48, 128},
       {{}, {0, 1}, {}},
       2,
       {{0}, {1, 2}, {}},
       {{0}, {1, 2}, {}},
       {{0}, {1, 2}, {}}},
      // axis = 2
      // [[0],[1,2],[]], [[],[0,1],[]] -> [[0],[1,2],[]], [[0],[1,2],[]],
      // [[0],[1,2],[]]
      {{2, 4, 128},
       {{0}, {1, 2}, {}},
       {2, 4, 128},
       {{}, {0, 1}, {}},
       2,
       {{0}, {1, 2}, {}},
       {{0}, {1, 2}, {}},
       {{0}, {1, 2}, {}}},
      // axis = 2
      // [[],[1,2],[]], [[],[0,1],[]] -> [[],[1,2],[]], [[],[1,2],[]],
      // [[],[1,2],[]]
      {{2, 4, 128},
       {{}, {1, 2}, {}},
       {2, 4, 128},
       {{}, {0, 1}, {}},
       2,
       {{}, {1, 2}, {}},
       {{}, {1, 2}, {}},
       {{}, {1, 2}, {}}},
      // axis = 1
      // [[0,1],[],[]], [[],[],[2]] -> [[0,1],[],[2]], [[0,1],[],[2]],
      // [[0,1],[],[2]]
      {{32, 48, 128},
       {{0, 1}, {}, {}},
       {32, 48, 128},
       {{}, {}, {2}},
       1,
       {{0, 1}, {}, {2}},
       {{0, 1}, {}, {2}},
       {{0, 1}, {}, {2}}},
      // Note: just for pass coverage ci: axis = 2
      // [[0],[0,1],[]], [[],[],[]] -> [[],[0,1],[]], [[],[0,1],[]],
      // [[],[0,1],[]]
      {{2, 4, 128},
       {{0}, {0, 1}, {}},
       {2, 4, 128},
       {{}, {}, {}},
       2,
       {{}, {0, 1}, {}},
       {{}, {0, 1}, {}},
       {{}, {0, 1}, {}}}};
  for (const auto& tc : test_cases) {
    TensorDistAttr out_dist_attr = TensorDistAttr();
    out_dist_attr.set_process_mesh(process_mesh);
    out_dist_attr.set_dims_mapping(tc.out_dims_mapping);
    out_dist_attr.set_dynamic_dims(
        std::vector<bool>(tc.out_shape.size(), false));
    phi::distributed::DistMetaTensor out = phi::distributed::DistMetaTensor(
        common::make_ddim(tc.out_shape), out_dist_attr);
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
        phi::distributed::SoftmaxGradInferSpmd(out, out_grad, tc.axis);
    EXPECT_EQ(backward_spmd_info.first.size(), static_cast<size_t>(2));
    EXPECT_EQ(backward_spmd_info.second.size(), static_cast<size_t>(1));
    check_multi_dims_mapping(backward_spmd_info.first[0],
                             tc.expected_out_dims_mapping);
    check_multi_dims_mapping(backward_spmd_info.first[1],
                             tc.expected_out_grad_dims_mapping);
    check_multi_dims_mapping(backward_spmd_info.second[0],
                             tc.expected_x_grad_dims_mapping);
  }
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
// [[0,1],[2]] [[2],[]]
