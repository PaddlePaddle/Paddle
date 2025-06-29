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

#include "glog/logging.h"
#include "test/cpp/auto_parallel/spmd_rule_test_util.h"
namespace paddle {
namespace distributed {
namespace auto_parallel {

ProcessMesh CreateProcessMesh() {
  std::vector<int64_t> mesh_shape = {2, 3};
  std::vector<int64_t> process_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  return ProcessMesh(mesh_shape, process_ids, dim_names);
}

phi::distributed::DistMetaTensor CreateDistMetaTensor(
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& dims_mapping,
    const ProcessMesh& process_mesh) {
  TensorDistAttr dist_attr;
  dist_attr.set_process_mesh(process_mesh);
  dist_attr.set_dims_mapping(dims_mapping);
  return phi::distributed::DistMetaTensor(phi::make_ddim(shape), dist_attr);
}

TEST(ExpandInferSpmd, Ctor) {
  ProcessMesh process_mesh = CreateProcessMesh();

  // Test case forward 1: Expand with shape {8, 2, 6, 1024, -1}
  auto x = CreateDistMetaTensor(
      {8, 2, 1, 1024, 128}, {0, -1, -1, 1, -1}, process_mesh);
  phi::IntArray shape = {8, 2, 6, 1024, -1};
  auto spmdinfo = ExpandInferSpmd(x, shape);
  EXPECT_EQ(get_dims_mapping(spmdinfo.first[0]),
            std::vector<int64_t>({0, -1, -1, 1, -1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo.second[0]),
            std::vector<int64_t>({0, -1, -1, 1, -1}));

  // Test case forward 2: Expand with shape {2, -1}
  auto x1 = CreateDistMetaTensor({8}, {1}, process_mesh);
  phi::IntArray shape1 = {2, -1};
  auto spmdinfo1 = ExpandInferSpmd(x1, shape1);
  EXPECT_EQ(get_dims_mapping(spmdinfo1.first[0]), std::vector<int64_t>({1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo1.second[0]),
            std::vector<int64_t>({-1, 1}));

  // Test case forward 3: Expand with shape {0, -1}
  auto x2 = CreateDistMetaTensor({8}, {1}, process_mesh);
  phi::IntArray shape2 = {0, -1};
  auto spmdinfo2 = ExpandInferSpmd(x2, shape2);
  EXPECT_EQ(get_dims_mapping(spmdinfo2.first[0]), std::vector<int64_t>({1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo2.second[0]),
            std::vector<int64_t>({-1, 1}));

  // Test case backward 1: ExpandGrad with shape {0, -1}
  auto x3 = CreateDistMetaTensor({8}, {1}, process_mesh);
  auto out3 = CreateDistMetaTensor({2, 8}, {-1, 1}, process_mesh);
  phi::IntArray shape3 = {0, -1};
  auto spmdinfo3 = ExpandGradInferSpmd(x3, out3, shape3);
  EXPECT_EQ(get_dims_mapping(spmdinfo3.first[0]), std::vector<int64_t>({1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo3.first[1]),
            std::vector<int64_t>({-1, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo3.second[0]), std::vector<int64_t>({1}));

  // Test case backward 2: ExpandGrad with shape {2, 2, -1}
  auto x4 = CreateDistMetaTensor({1, 8}, {-1, 1}, process_mesh);
  auto out4 = CreateDistMetaTensor({2, 2, 8}, {-1, -1, 1}, process_mesh);
  phi::IntArray shape4 = {2, 2, -1};
  auto spmdinfo4 = ExpandGradInferSpmd(x4, out4, shape4);
  EXPECT_EQ(get_dims_mapping(spmdinfo4.first[0]),
            std::vector<int64_t>({-1, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo4.first[1]),
            std::vector<int64_t>({-1, -1, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo4.second[0]),
            std::vector<int64_t>({-1, 1}));

  // Test case backward 3: ExpandGrad with shape {2, 2, -1}
  auto x5 = CreateDistMetaTensor({1, 8}, {-1, 1}, process_mesh);
  auto out5 = CreateDistMetaTensor({2, 2, 8}, {-1, 0, 1}, process_mesh);
  phi::IntArray shape5 = {2, 2, -1};
  auto spmdinfo5 = ExpandGradInferSpmd(x5, out5, shape5);
  EXPECT_EQ(get_dims_mapping(spmdinfo5.first[0]),
            std::vector<int64_t>({-1, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo5.first[1]),
            std::vector<int64_t>({-1, 0, 1}));
  EXPECT_EQ(get_dims_mapping(spmdinfo5.second[0]),
            std::vector<int64_t>({-1, 1}));
  EXPECT_EQ(get_partial_dims(spmdinfo5.second[0]), std::set<int64_t>({0}));
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
