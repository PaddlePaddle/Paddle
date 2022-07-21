/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/distributed/auto_parallel/device_mesh.h"
#include <iostream>
#include "gtest/gtest.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

TEST(DeviceMesh, Ctor) {
  std::vector<int64_t> shape = {2, 3};
  std::vector<int64_t> device_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  std::string device_type = "GPU";
  int64_t size = shape[0] * shape[1];
  std::vector<Device> devices(size);
  for (int64_t i = 0; i < shape[0]; ++i) {
    for (int64_t j = 0; j < shape[1]; ++j) {
      int64_t idx = i * shape[1] + j;
      devices[idx].global_id = i * shape[1] + j;
      devices[idx].local_id = j;
      devices[idx].type = device_type;
    }
  }
  std::unordered_map<std::int64_t, std::vector<Link>> links;
  for (int64_t i = 0; i < size; ++i) {
    for (int64_t j = 0; j < size; ++j) {
      Link link;
      link.source_id = i;
      link.target_id = j;
      links[i].push_back(link);
    }
  }
  DeviceMesh device_mesh(
      shape, device_ids, dim_names, device_type, devices, links);
  EXPECT_EQ(device_mesh.shape(), shape);
  EXPECT_EQ(device_mesh.device_ids(), device_ids);
  EXPECT_EQ(device_mesh.dim_names()[0], "x");
  EXPECT_EQ(device_mesh.dim_names()[1], "y");
  EXPECT_EQ(device_mesh.device_type(), device_type);
  EXPECT_EQ(device_mesh.devices(), devices);
  EXPECT_EQ(device_mesh.links(), links);
  EXPECT_EQ(device_mesh.size(), size);
  EXPECT_EQ(device_mesh.ndim(), static_cast<int64_t>(shape.size()));
  EXPECT_EQ(device_mesh.dim_size(0), shape[0]);
  EXPECT_EQ(device_mesh.dim_size(-1), shape[1]);
  EXPECT_EQ(device_mesh.dim_size("x"), shape[0]);
  EXPECT_EQ(device_mesh.dim_size("y"), shape[1]);
  // std::cout << device_mesh << std::endl;
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
