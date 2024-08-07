/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

TEST(Tile, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<int64_t> shape = {6, 8, 10};
  std::vector<int64_t> dims_mapping = {0, -1, 1};

  TensorDistAttr t_dist_attr = TensorDistAttr();
  t_dist_attr.set_process_mesh(process_mesh);
  t_dist_attr.set_dims_mapping(dims_mapping);
  t_dist_attr.set_dynamic_dims({false, false, false});
  phi::distributed::DistMetaTensor x =
      phi::distributed::DistMetaTensor(common::make_ddim(shape), t_dist_attr);
  std::vector<int64_t> repeat_times = {2, 2, 1, 1};
  // test forward
  phi::distributed::SpmdInfo forward_spmd_info =
      phi::distributed::TileInferSpmd(x, repeat_times);
  EXPECT_EQ(forward_spmd_info.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(forward_spmd_info.second.size(), static_cast<size_t>(1));
  check_dim_mapping(forward_spmd_info.first[0], {-1, -1, 1});
  check_dim_mapping(forward_spmd_info.second[0], {-1, -1, -1, 1});
  check_partial_dims(forward_spmd_info.second[0], {});

  // test backward
  auto out_grad_dist_attr =
      PADDLE_GET_CONST(TensorDistAttr, forward_spmd_info.second[0]);
  out_grad_dist_attr.set_dims_mapping({0, -1, -1, 1});
  phi::distributed::DistMetaTensor out_grad = phi::distributed::DistMetaTensor(
      common::make_ddim({2, 12, 8, 10}), out_grad_dist_attr);
  phi::distributed::SpmdInfo backward_spmd_info =
      TileGradInferSpmd(x, out_grad, repeat_times);
  EXPECT_EQ(backward_spmd_info.first.size(), static_cast<size_t>(2));
  EXPECT_EQ(backward_spmd_info.second.size(), static_cast<size_t>(1));
  check_dim_mapping(backward_spmd_info.first[0], {-1, -1, 1});
  check_dim_mapping(backward_spmd_info.first[1], {0, -1, -1, 1});
  check_dim_mapping(backward_spmd_info.second[0], {-1, -1, 1});
  check_partial_dims(backward_spmd_info.second[0], {0});
}
}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
