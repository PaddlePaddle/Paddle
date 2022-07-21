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

#include <algorithm>

#include "paddle/fluid/distributed/auto_parallel/dist_mapper.h"
#include "paddle/fluid/distributed/auto_parallel/utils.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

void DistributedMapper::set_process_id_to_device_ids(
    const std::map<int64_t, std::pair<int64_t, std::vector<int64_t>>>&
        process_id_to_device_ids) {
  std::vector<int64_t> device_mesh_ids;
  for (const auto& item : device_meshes_) {
    device_mesh_ids.push_back(item.first);
  }
  for (const auto& item : process_id_to_device_ids) {
    PADDLE_ENFORCE_GE(
        item.first,
        0,
        platform::errors::InvalidArgument(
            "The process id %d must be greater than or equal to 0.",
            item.first));
    int64_t device_mesh_id = item.second.first;
    const std::vector<int64_t>& device_ids = item.second.second;
    PADDLE_ENFORCE_EQ(
        device_meshes_.count(device_mesh_id),
        1,
        platform::errors::InvalidArgument(
            "Cannot find the device mesh %d in device_mesh ids [%s].",
            device_mesh_id,
            str_join(device_mesh_ids)));
    PADDLE_ENFORCE_EQ(
        has_duplicates(device_ids),
        false,
        platform::errors::InvalidArgument(
            "The mapped device ids [%s] of process_mesh %d must be unique.",
            str_join(device_ids),
            item.first));
    const DeviceMesh& device_mesh = *device_meshes_[device_mesh_id];
    const std::vector<int64_t> cur_device_ids = device_mesh.device_ids();
    for (int64_t device_id : device_ids) {
      bool found =
          std::find(cur_device_ids.begin(), cur_device_ids.end(), device_id) !=
          cur_device_ids.end();
      PADDLE_ENFORCE_EQ(
          found,
          true,
          platform::errors::InvalidArgument(
              "The device id %d cannot be find in the device mesh [%s].",
              device_id,
              str_join(cur_device_ids)));
    }
  }
  process_id_to_device_ids_ = process_id_to_device_ids;
}

std::string DistributedMapper::to_string() const {
  std::string mapper_str = "{device_meshes:[";
  for (const auto& item : device_meshes_) {
    mapper_str += item.second->to_string() + ",";
  }
  mapper_str.replace(mapper_str.size() - 1, 1, "]");

  mapper_str += "process_id_to_device_ids:[";
  for (const auto& item : process_id_to_device_ids_) {
    mapper_str += "{";
    mapper_str += "process_id:" + std::to_string(item.first) + ", device_ids:[";
    for (const auto& device_id : item.second.second) {
      mapper_str += "{" + std::to_string(item.second.first) + "," +
                    std::to_string(device_id) + "},";
    }
    mapper_str.replace(mapper_str.size() - 1, 1, "]");
    mapper_str += "},";
  }
  mapper_str.replace(mapper_str.size() - 1, 1, "]");
  mapper_str += "}";
  return mapper_str;
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
