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

#include <iostream>
#include <sstream>

#include "paddle/phi/core/distributed/auto_parallel/device_mesh.h"
#include "paddle/phi/core/distributed/auto_parallel/proto_helper.h"

#include "gtest/gtest.h"

namespace phi {
namespace distributed {
namespace auto_parallel {

TEST(DeviceMesh, Ctor) {
  std::vector<int64_t> shape = {2, 3};
  std::vector<int64_t> device_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  std::string device_type = "GPU";
  int64_t size = shape[0] * shape[1];

  DeviceMesh device_mesh("mesh", shape, device_ids, dim_names);
  for (int64_t i = 0; i < shape[0]; ++i) {
    for (int64_t j = 0; j < shape[1]; ++j) {
      int64_t global_id = i * shape[1] + j;
      int64_t local_id = j;
      int64_t machine_id = i;
      device_mesh.add_device(
          Device(global_id, local_id, machine_id, device_type));
    }
  }
  for (int64_t i = 0; i < size; ++i) {
    for (int64_t j = 0; j < size; ++j) {
      device_mesh.add_link(Link(i, j, "NVL"));
    }
  }

  EXPECT_EQ(device_mesh.name(), "mesh");
  EXPECT_EQ(device_mesh.shape(), shape);
  EXPECT_EQ(device_mesh.device_ids(), device_ids);
  EXPECT_EQ(device_mesh.dim_names()[0], "x");
  EXPECT_EQ(device_mesh.dim_names()[1], "y");
  EXPECT_EQ(device_mesh.device_type(), device_type);
  EXPECT_EQ(device_mesh.size(), size);
  EXPECT_EQ(device_mesh.ndim(), static_cast<int64_t>(shape.size()));
  EXPECT_EQ(device_mesh.dim_size(0), shape[0]);
  EXPECT_EQ(device_mesh.dim_size(-1), shape[1]);
  EXPECT_EQ(device_mesh.dim_size("x"), shape[0]);
  EXPECT_EQ(device_mesh.dim_size("y"), shape[1]);
  EXPECT_EQ(device_mesh.empty(), false);
  EXPECT_EQ(device_mesh.contains(0), true);
  EXPECT_EQ(device_mesh.contains(6), false);
  EXPECT_EQ(device_mesh.device(3).global_id(), 3);
  EXPECT_EQ(device_mesh.device(3).local_id(), 0);
  EXPECT_EQ(device_mesh.device(3).machine_id(), 1);
  EXPECT_EQ(device_mesh.device(3).type(), "GPU");
  EXPECT_EQ(device_mesh.link(3, 4).source_id(), 3);
  EXPECT_EQ(device_mesh.link(3, 4).target_id(), 4);
  EXPECT_EQ(device_mesh.link(3, 4).type(), "NVL");
  for (int64_t i = 0; i < shape[0]; ++i) {
    for (int64_t j = 0; j < shape[1]; ++j) {
      int64_t global_id = i * shape[1] + j;
      int64_t local_id = j;
      int64_t machine_id = i;
      auto device = device_mesh.devices().at(global_id);
      EXPECT_EQ(device, Device(global_id, local_id, machine_id, device_type));
    }
  }
  for (int64_t i = 0; i < size; ++i) {
    for (int64_t j = 0; j < size; ++j) {
      EXPECT_EQ(device_mesh.links().at(i).at(j), Link(i, j, "NVL"));
    }
  }
  std::stringstream sstream;
  sstream << device_mesh;
  EXPECT_EQ(sstream.str(), device_mesh.to_string());
  auto proto = phi::distributed::to_proto(device_mesh);
  DeviceMesh new_device_mesh = DeviceMesh::from_proto(proto);
  EXPECT_EQ(device_mesh, new_device_mesh);
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace phi
