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

TEST(Reshape, Ctor) {
  std::vector<int64_t> mesh_shape = {2, 2};
  std::vector<int64_t> process_ids = {0, 1, 2, 3};
  std::vector<std::string> dim_names = {"x", "y"};
  ProcessMesh process_mesh(mesh_shape, process_ids, dim_names);

  std::vector<int64_t> shape = {4, 6, 8};
  // [4, 6, 8] -> [2, 12, 8]
  std::vector<std::vector<int64_t>> dims_mapping = {{0, 1}, {}, {}};

  TensorDistAttr t_dist_attr = TensorDistAttr();
  t_dist_attr.set_process_mesh(process_mesh);
  t_dist_attr.set_dims_mapping(dims_mapping);
  t_dist_attr.set_dynamic_dims({false, false, false});
  phi::distributed::DistMetaTensor x =
      phi::distributed::DistMetaTensor(common::make_ddim(shape), t_dist_attr);
  std::vector<int64_t> target_shape = {2, 12, 8};
  // test forward
  phi::distributed::SpmdInfo forward_spmd_info =
      phi::distributed::ReshapeInferSpmd(x, target_shape);
  EXPECT_EQ(forward_spmd_info.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(forward_spmd_info.second.size(), static_cast<size_t>(1));
  check_multi_dims_mapping(forward_spmd_info.first[0], {{}, {}, {}});
  check_multi_dims_mapping(forward_spmd_info.second[0], {{}, {}, {}});

  // [4, 6, 8] -> [24, 2, 4]
  target_shape = {24, 2, 4};
  dims_mapping = {{0, 1}, {}, {}};
  t_dist_attr.set_dims_mapping(dims_mapping);
  x = phi::distributed::DistMetaTensor(common::make_ddim(shape), t_dist_attr);
  forward_spmd_info = phi::distributed::ReshapeInferSpmd(x, target_shape);
  EXPECT_EQ(forward_spmd_info.first.size(), static_cast<size_t>(1));
  EXPECT_EQ(forward_spmd_info.second.size(), static_cast<size_t>(1));
  check_multi_dims_mapping(forward_spmd_info.first[0], {{0, 1}, {}, {}});
  check_multi_dims_mapping(forward_spmd_info.second[0], {{0, 1}, {}, {}});
}
}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
