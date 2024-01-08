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

#include "paddle/phi/core/distributed/auto_parallel/dist_mapper.h"
#include <map>
#include <sstream>
#include "gtest/gtest.h"
#include "paddle/phi/core/distributed/auto_parallel/proto_helper.h"

namespace phi {
namespace distributed {
namespace auto_parallel {

TEST(DistributedMapper, Ctor) {
  std::vector<int64_t> shape = {2, 3};
  std::vector<int64_t> device_ids = {0, 1, 2, 3, 4, 5};
  std::vector<std::string> dim_names = {"x", "y"};
  std::string device_type = "GPU";
  int64_t size = shape[0] * shape[1];

  DeviceMesh device_mesh("device_mesh", shape, device_ids, dim_names);
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

  DistributedMapper dist_mapper;
  dist_mapper.add_device_mesh(device_mesh);
  std::map<int64_t, std::pair<std::string, std::vector<int64_t>>>
      process_id_to_device_ids;
  process_id_to_device_ids[0] = {"device_mesh", {5}};
  process_id_to_device_ids[1] = {"device_mesh", {4}};
  process_id_to_device_ids[2] = {"device_mesh", {3}};
  process_id_to_device_ids[3] = {"device_mesh", {2}};
  process_id_to_device_ids[4] = {"device_mesh", {1}};
  process_id_to_device_ids[5] = {"device_mesh", {0}};
  dist_mapper.set_process_id_to_device_ids(process_id_to_device_ids);

  EXPECT_EQ(dist_mapper.device_meshes().at("device_mesh"), device_mesh);
  EXPECT_EQ(dist_mapper.device_mesh("device_mesh"), device_mesh);
  EXPECT_EQ(dist_mapper.process_id_to_device_ids(), process_id_to_device_ids);
  std::stringstream sstream;
  sstream << dist_mapper;
  EXPECT_EQ(sstream.str(), dist_mapper.to_string());
  auto proto = phi::distributed::to_proto(dist_mapper);
  DistributedMapper new_dist_mapper = DistributedMapper::from_proto(proto);
  EXPECT_EQ(dist_mapper, new_dist_mapper);
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace phi
