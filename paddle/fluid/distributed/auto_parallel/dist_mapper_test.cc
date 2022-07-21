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

#include "paddle/fluid/distributed/auto_parallel/dist_mapper.h"
#include <iostream>
#include <map>
#include "gtest/gtest.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

TEST(DistributedMapper, Ctor) {
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
  DistributedMapper dist_mapper;
  dist_mapper.add_device_mesh(device_mesh);
  std::map<int64_t, std::pair<int64_t, std::vector<int64_t>>>
      process_id_to_device_ids;
  process_id_to_device_ids[0] = {0, {5}};
  process_id_to_device_ids[1] = {0, {4}};
  process_id_to_device_ids[2] = {0, {3}};
  process_id_to_device_ids[3] = {0, {2}};
  process_id_to_device_ids[4] = {0, {1}};
  process_id_to_device_ids[5] = {0, {0}};
  dist_mapper.set_process_id_to_device_ids(process_id_to_device_ids);
  EXPECT_EQ(*dist_mapper.device_meshes()[0], device_mesh);
  EXPECT_EQ(dist_mapper.process_id_to_device_ids(), process_id_to_device_ids);
  std::cout << dist_mapper << std::endl;
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
