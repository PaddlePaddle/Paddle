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

struct MatmulTestCase {
  // input
  std::vector<int64_t> x_shape;
  std::vector<std::vector<int64_t>> x_dims_mapping;

  std::vector<int64_t> y_shape;
  std::vector<std::vector<int64_t>> y_dims_mapping;

  // attribute
  bool trans_x;
  bool trans_y;

  // output
  std::vector<std::vector<int64_t>> expected_x_dims_mapping;
  std::vector<std::vector<int64_t>> expected_y_dims_mapping;
  std::vector<std::vector<int64_t>> expected_out_dims_mapping;

  std::set<int64_t> partial_dims;
};

struct MatmulGradTestCase {
  // input
  std::vector<int64_t> x_shape;
  std::vector<std::vector<int64_t>> x_dims_mapping;

  std::vector<int64_t> y_shape;
  std::vector<std::vector<int64_t>> y_dims_mapping;

  std::vector<int64_t> out_grad_shape;
  std::vector<std::vector<int64_t>> out_grad_dims_mapping;

  // attribute
  bool trans_x;
  bool trans_y;

  // output
  std::vector<std::vector<int64_t>> expected_x_dims_mapping;
  std::vector<std::vector<int64_t>> expected_y_dims_mapping;
  std::vector<std::vector<int64_t>> expected_out_grad_dims_mapping;

  std::vector<std::vector<int64_t>> expected_x_grad_dims_mapping;
  std::vector<std::vector<int64_t>> expected_y_grad_dims_mapping;

  std::set<int64_t> x_grad_partial_dims;
  std::set<int64_t> y_grad_partial_dims;
};

TEST(MatmulInferSpmd, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::string> dim_names = {"x", "y", "z"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<MatmulTestCase> test_cases = {
      // [64, 32], [32, 48], trans_x=false, trans_y=false
      // [[0,1], []] ,[[],[2]] -> [[0,1], []] ,[[],[2]],[[0,1],[2]]
      {{64, 32},
       {{0, 1}, {}},
       {32, 48},
       {{}, {2}},
       false,
       false,
       {{0, 1}, {}},
       {{}, {2}},
       {{0, 1}, {2}},
       {}},

      // [64, 32], [32, 48], trans_x=false, trans_y=false
      // [[0,1], [2]] ,[[],[]] -> [[0,1], [2]] ,[[2],[]],[[0,1],[]], partial: 2
      {{64, 32},
       {{0, 1}, {2}},
       {32, 48},
       {{}, {}},
       false,
       false,
       {{0, 1}, {2}},
       {{2}, {}},
       {{0, 1}, {}},
       {2}},

      // [64, 32], [32, 48], trans_x=false, trans_y=false
      // [[], []] ,[[0,1],[2]] -> [[],[0,1]] ,[[0,1],[2],[[],[2]], partial:
      // {0,1}
      {{64, 32},
       {{}, {}},
       {32, 48},
       {{0, 1}, {2}},
       false,
       false,
       {{}, {0, 1}},
       {{0, 1}, {2}},
       {{}, {2}},
       {0, 1}},

      // [64, 32], [32, 48], trans_x=false, trans_y=false
      // [[0], [1]] ,[[2],[0]] -> [[0], [1,2]] ,[[1,2],[]],[[0],[]], partial:
      // {1,2}
      {{64, 32},
       {{0}, {1}},
       {32, 48},
       {{2}, {0}},
       false,
       false,
       {{0}, {1, 2}},
       {{1, 2}, {}},
       {{0}, {}},
       {1, 2}},

      // [64, 32], [32, 48], trans_x=false, trans_y=false
      // [[0,1], [2]] ,[[0],[]] -> [[0,1], [2]] ,[[2],[]],[[0,1],[]], partial: 2
      {{64, 32},
       {{0, 1}, {2}},
       {32, 48},
       {{0}, {}},
       false,
       false,
       {{0, 1}, {2}},
       {{2}, {}},
       {{0, 1}, {}},
       {2}},

      // [512, 48, 64, 32], [1, 32, 48], trans_x=false, trans_y=false
      // [[0,1],[2],[],[]] ,[[],[],[]] -> [[0,1],[2],[],[]]
      // ,[[],[],[]],[[0,1],[2],[],[]],
      // partial: {}
      {{512, 48, 64, 32},
       {{0, 1}, {2}, {}, {}},
       {1, 32, 48},
       {{}, {}, {}},
       false,
       false,
       {{0, 1}, {2}, {}, {}},
       {{}, {}, {}},
       {{0, 1}, {2}, {}, {}},
       {}},

      // [512, 48, 32, 64], [1, 32, 48], trans_x=true, trans_y=false
      // [[0],[],[1,2],[]] ,[[],[],[2]] -> [[0],[],[1],[]]
      // ,[[],[1],[2]],[[0],[],[],[2]],
      // partial: {1}
      {{512, 48, 32, 64},
       {{0}, {}, {1, 2}, {}},
       {1, 32, 48},
       {{}, {}, {2}},
       true,
       false,
       {{0}, {}, {1}, {}},
       {{}, {1}, {2}},
       {{0}, {}, {}, {2}},
       {1}},

      // [512, 48, 64, 32], [1, 48, 32], trans_x=false, trans_y=true
      // [[0],[],[1,2],[]] ,[[],[0],[]] -> [[0],[],[1,2],[]]
      // ,[[],[],[]],[[0],[],[1,2],[]],
      // partial: {}
      {{512, 48, 64, 32},
       {{0}, {}, {1, 2}, {}},
       {1, 48, 32},
       {{}, {0}, {}},
       false,
       true,
       {{0}, {}, {1, 2}, {}},
       {{}, {}, {}},
       {{0}, {}, {1, 2}, {}},
       {}},

      // [512, 48, 32, 64], [1, 48, 32], trans_x=true, trans_y=true
      // [[],[],[0,1],[2]] ,[[],[0,1],[2]] -> [[],[],[],[2]]
      // ,[[],[0,1],[]],[[],[],[2],[0,1]],
      // partial: {}
      {{512, 48, 32, 64},
       {{}, {}, {0, 1}, {2}},
       {1, 48, 32},
       {{}, {0, 1}, {2}},
       true,
       true,
       {{}, {}, {}, {2}},
       {{}, {0, 1}, {}},
       {{}, {}, {2}, {0, 1}},
       {}},
  };
  for (const auto& tc : test_cases) {
    TensorDistAttr x_dist_attr = TensorDistAttr();
    x_dist_attr.set_process_mesh(process_mesh);
    x_dist_attr.set_dims_mapping(tc.x_dims_mapping);
    x_dist_attr.set_dynamic_dims(std::vector<bool>(tc.x_shape.size(), false));
    phi::distributed::DistMetaTensor x = phi::distributed::DistMetaTensor(
        common::make_ddim(tc.x_shape), x_dist_attr);

    TensorDistAttr y_dist_attr = TensorDistAttr();
    y_dist_attr.set_process_mesh(process_mesh);
    y_dist_attr.set_dims_mapping(tc.y_dims_mapping);
    y_dist_attr.set_dynamic_dims(std::vector<bool>(tc.y_shape.size(), false));
    phi::distributed::DistMetaTensor y = phi::distributed::DistMetaTensor(
        common::make_ddim(tc.y_shape), y_dist_attr);

    // test forward
    phi::distributed::SpmdInfo forward_spmd_info =
        phi::distributed::MatmulInferSpmd(x, y, tc.trans_x, tc.trans_y);
    EXPECT_EQ(forward_spmd_info.first.size(), static_cast<size_t>(2));
    EXPECT_EQ(forward_spmd_info.second.size(), static_cast<size_t>(1));
    check_multi_dims_mapping(forward_spmd_info.first[0],
                             tc.expected_x_dims_mapping);
    check_multi_dims_mapping(forward_spmd_info.first[1],
                             tc.expected_y_dims_mapping);
    check_multi_dims_mapping(forward_spmd_info.second[0],
                             tc.expected_out_dims_mapping);
    if (!tc.partial_dims.empty()) {
      EXPECT_EQ(is_partial(forward_spmd_info.second[0]), true);
      check_partial_dims(forward_spmd_info.second[0], tc.partial_dims);
    }
  }
}

TEST(MatmulGradInferSpmd, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<std::string> dim_names = {"x", "y", "z"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<MatmulGradTestCase> test_cases = {
      // [64, 32], [32, 48], [64,48], trans_x=false, trans_y=false
      // [[0,1], []] ,[[],[2]], [[0,1],[2]] -> [[0,1], []]
      // ,[[],[2]],[[0,1],[2]], [[0,1],[]], [[],[2]], x_partial: {2}, y_partial:
      // {0,1}
      {{64, 32},
       {{0, 1}, {}},
       {32, 48},
       {{}, {2}},
       {64, 48},
       {{0, 1}, {2}},
       false,
       false,
       {{0, 1}, {}},
       {{}, {2}},
       {{0, 1}, {2}},
       {{0, 1}, {}},
       {{}, {2}},
       {2},
       {0, 1}},
      // [1024,512,64,32], [1,32,48], [1024,512,64,48], trans_x=false,
      // trans_y=false
      // [[0],[],[1,2],[]] ,[[],[],[2]], [[0],[],[1,2],[]] -> [[0],[],[1,2],[]]
      // ,[[],[],[]], [[0],[],[1,2],[]], [[0],[],[1,2],[]], [[],[],[]],
      // x_grad_partial: {}, y_grad_partial: {0,1,2}
      {{1024, 512, 64, 32},
       {{0}, {}, {1, 2}, {}},
       {1, 32, 48},
       {{}, {}, {2}},
       {1024, 512, 64, 48},
       {{0}, {}, {1, 2}, {}},
       false,
       false,
       {{0}, {}, {1, 2}, {}},
       {{}, {}, {}},
       {{0}, {}, {1, 2}, {}},
       {{0}, {}, {1, 2}, {}},
       {{}, {}, {}},
       {},
       {0, 1, 2}},
      // [1024,512,64,32], [1,32,48], [1024,512,64,48], trans_x=false,
      // trans_y=false
      // [[],[0],[1,2],[]] ,[[],[],[2]], [[],[0],[1,2],[]] -> [[],[0],[1,2],[]]
      // ,[[],[],[]], [[],[0],[1,2],[]], [[],[0],[1,2],[]], [[],[],[]],
      // x_grad_partial: {}, y_grad_partial: {0,1,2}
      {{1024, 512, 64, 32},
       {{}, {0}, {1, 2}, {}},
       {1, 32, 48},
       {{}, {}, {2}},
       {1024, 512, 64, 48},
       {{}, {0}, {1, 2}, {}},
       false,
       false,
       {{}, {0}, {1, 2}, {}},
       {{}, {}, {}},
       {{}, {0}, {1, 2}, {}},
       {{}, {0}, {1, 2}, {}},
       {{}, {}, {}},
       {},
       {0, 1, 2}},
      // [1024,512,32,64], [1,32,48], [1024,512,64,48], trans_x=true,
      // trans_y=false
      // [[],[0],[1,2],[]] ,[[],[],[2]], [[],[0],[],[2]] -> [[],[0],[1],[]]
      // ,[[],[1],[2]], [[],[0],[],[2]], [[],[0],[1],[]], [[],[1],[2]],
      // x_grad_partial: {2}, y_grad_partial: {0}
      {{1024, 512, 32, 64},
       {{}, {0}, {1, 2}, {}},
       {1, 32, 48},
       {{}, {}, {2}},
       {1024, 512, 64, 48},
       {{}, {0}, {}, {2}},
       true,
       false,
       {{}, {0}, {1}, {}},
       {{}, {1}, {2}},
       {{}, {0}, {}, {2}},
       {{}, {0}, {1}, {}},
       {{}, {1}, {2}},
       {2},
       {0}},
      // [1024,512,32,64], [1,48,32], [1024,512,64,48], trans_x=true,
      // trans_y=true
      // [[],[],[1,2],[]] ,[[],[],[0]], [[],[],[],[]] -> [[],[],[0,1,2],[]]
      // ,[[],[],[0,1,2]], [[],[],[],[]], [[],[],[0,1,2],[]], [[],[],[0,1,2]],
      // x_grad_partial: {}, y_grad_partial: {}
      {{1024, 512, 32, 64},
       {{}, {}, {1, 2}, {}},
       {1, 48, 32},
       {{}, {}, {0}},
       {1024, 512, 64, 48},
       {{}, {}, {}, {}},
       true,
       true,
       {{}, {}, {1, 2, 0}, {}},
       {{}, {}, {1, 2, 0}},
       {{}, {}, {}, {}},
       {{}, {}, {1, 2, 0}, {}},
       {{}, {}, {1, 2, 0}},
       {},
       {}},
      // [1024,512,64,32], [1,48,32], [1024,512,64,48], trans_x=false,
      // trans_y=true
      // [[],[],[0],[1,2]] ,[[],[],[0]], [[],[],[0],[]] -> [[],[],[0],[1,2]]
      // ,[[],[],[1,2]], [[],[],[0],[]], [[],[],[0],[1,2]],
      // [[],[],[1,2]],
      // x_grad_partial: {}, y_grad_partial: {0}
      {{1024, 512, 64, 32},
       {{}, {}, {0}, {1, 2}},
       {1, 48, 32},
       {{}, {}, {0}},
       {1024, 512, 64, 48},
       {{}, {}, {0}, {}},
       false,
       true,
       {{}, {}, {0}, {1, 2}},
       {{}, {}, {1, 2}},
       {{}, {}, {0}, {}},
       {{}, {}, {0}, {1, 2}},
       {{}, {}, {1, 2}},
       {},
       {0}}};
  for (const auto& tc : test_cases) {
    TensorDistAttr x_dist_attr = TensorDistAttr();
    x_dist_attr.set_process_mesh(process_mesh);
    x_dist_attr.set_dims_mapping(tc.x_dims_mapping);
    x_dist_attr.set_dynamic_dims(std::vector<bool>(tc.x_shape.size(), false));
    phi::distributed::DistMetaTensor x = phi::distributed::DistMetaTensor(
        common::make_ddim(tc.x_shape), x_dist_attr);

    TensorDistAttr y_dist_attr = TensorDistAttr();
    y_dist_attr.set_process_mesh(process_mesh);
    y_dist_attr.set_dims_mapping(tc.y_dims_mapping);
    y_dist_attr.set_dynamic_dims(std::vector<bool>(tc.y_shape.size(), false));
    phi::distributed::DistMetaTensor y = phi::distributed::DistMetaTensor(
        common::make_ddim(tc.y_shape), y_dist_attr);

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
        phi::distributed::MatmulGradInferSpmd(
            x, y, out_grad, tc.trans_x, tc.trans_y);
    EXPECT_EQ(backward_spmd_info.first.size(), static_cast<size_t>(3));
    EXPECT_EQ(backward_spmd_info.second.size(), static_cast<size_t>(2));
    check_multi_dims_mapping(backward_spmd_info.first[0],
                             tc.expected_x_dims_mapping);
    check_multi_dims_mapping(backward_spmd_info.first[1],
                             tc.expected_y_dims_mapping);
    check_multi_dims_mapping(backward_spmd_info.first[2],
                             tc.expected_out_grad_dims_mapping);
    check_multi_dims_mapping(backward_spmd_info.second[0],
                             tc.expected_x_grad_dims_mapping);
    if (!tc.x_grad_partial_dims.empty()) {
      EXPECT_EQ(is_partial(backward_spmd_info.second[0]), true);
      check_partial_dims(backward_spmd_info.second[0], tc.x_grad_partial_dims);
    }
    check_multi_dims_mapping(backward_spmd_info.second[1],
                             tc.expected_y_grad_dims_mapping);
    if (!tc.y_grad_partial_dims.empty()) {
      EXPECT_EQ(is_partial(backward_spmd_info.second[1]), true);
      check_partial_dims(backward_spmd_info.second[1], tc.y_grad_partial_dims);
    }
  }
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
