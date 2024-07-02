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

#include "paddle/phi/core/distributed/auto_parallel/dist_mapper.h"
#include "paddle/phi/core/distributed/auto_parallel/proto_helper.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"

namespace phi::distributed::auto_parallel {

void DistributedMapper::set_process_id_to_device_ids(
    const std::map<int64_t, std::pair<std::string, std::vector<int64_t>>>&
        process_id_to_device_ids) {
  std::vector<std::string> device_mesh_names;
  device_mesh_names.reserve(device_meshes_.size());
  for (const auto& item : device_meshes_) {
    device_mesh_names.push_back(item.first);
  }
  for (const auto& item : process_id_to_device_ids) {
    PADDLE_ENFORCE_GE(
        item.first,
        0,
        errors::InvalidArgument(
            "The process id %d must be greater than or equal to 0.",
            item.first));
    std::string device_mesh_name = item.second.first;
    const std::vector<int64_t>& device_ids = item.second.second;
    PADDLE_ENFORCE_EQ(
        device_meshes_.count(device_mesh_name),
        1,
        errors::InvalidArgument(
            "Cannot find the device mesh %d in device_mesh ids [%s].",
            device_mesh_name,
            str_join(device_mesh_names)));
    PADDLE_ENFORCE_EQ(
        has_duplicates(device_ids),
        false,
        errors::InvalidArgument(
            "The mapped device ids [%s] of process_mesh %d must be unique.",
            str_join(device_ids),
            item.first));
    const DeviceMesh& device_mesh = device_meshes_[device_mesh_name];
    const std::vector<int64_t> cur_device_ids = device_mesh.device_ids();
    for (int64_t device_id : device_ids) {
      bool found =
          std::find(cur_device_ids.begin(), cur_device_ids.end(), device_id) !=
          cur_device_ids.end();
      PADDLE_ENFORCE_EQ(
          found,
          true,
          errors::InvalidArgument(
              "The device id %d cannot be find in the device mesh [%s].",
              device_id,
              str_join(cur_device_ids)));
    }
  }
  process_id_to_device_ids_ = process_id_to_device_ids;
}

DistributedMapper DistributedMapper::from_proto(
    const DistributedMapperProto& proto) {
  DistributedMapper dist_mapper;
  for (int i = 0; i < proto.device_meshes_size(); ++i) {
    dist_mapper.device_meshes_[proto.device_meshes(i).name()] =
        DeviceMesh::from_proto(proto.device_meshes(i));
  }
  for (int i = 0; i < proto.process_id_to_device_ids_size(); ++i) {
    int64_t process_id = proto.process_id_to_device_ids(i).process_id();
    std::string device_mesh_name =
        proto.process_id_to_device_ids(i).device_mesh_name();
    std::vector<int64_t> device_ids;
    int num_devices = proto.process_id_to_device_ids(i).device_ids_size();
    device_ids.reserve(num_devices);
    for (int j = 0; j < num_devices; ++j) {
      device_ids.push_back(proto.process_id_to_device_ids(i).device_ids(j));
    }
    dist_mapper.process_id_to_device_ids_[process_id].first = device_mesh_name;
    dist_mapper.process_id_to_device_ids_[process_id].second = device_ids;
  }
  return dist_mapper;
}

void DistributedMapper::to_proto(DistributedMapperProto* proto) const {
  for (const auto& item : device_meshes_) {
    proto->mutable_device_meshes()->Add()->CopyFrom(
        phi::distributed::to_proto(item.second));
  }
  for (const auto& outer : process_id_to_device_ids_) {
    auto proto_item = proto->mutable_process_id_to_device_ids()->Add();
    proto_item->set_process_id(outer.first);
    proto_item->set_device_mesh_name(outer.second.first);
    for (const auto& inner : outer.second.second) {
      proto_item->add_device_ids(inner);
    }
  }
}

std::string DistributedMapper::to_string() const {
  std::string mapper_str = "{device_meshes: [";
  for (const auto& item : device_meshes_) {
    mapper_str += item.second.to_string() + ", ";
  }
  mapper_str.replace(mapper_str.size() - 2, 2, "]");

  mapper_str += "\nprocess_id_to_device_ids: [";
  for (const auto& item : process_id_to_device_ids_) {
    mapper_str += "{";
    mapper_str +=
        "process_id: " + std::to_string(item.first) + ", device_ids: [";
    for (const auto& device_id : item.second.second) {
      mapper_str +=
          "{" + item.second.first + ", " + std::to_string(device_id) + "}, ";
    }
    mapper_str.replace(mapper_str.size() - 2, 2, "]");
    mapper_str += "}, ";
  }
  mapper_str.replace(mapper_str.size() - 2, 2, "]");
  mapper_str += "}";
  return mapper_str;
}

bool operator==(const DistributedMapper& lhs, const DistributedMapper& rhs) {
  if (lhs.device_meshes() != rhs.device_meshes()) {
    return false;
  }
  if (lhs.process_id_to_device_ids() != rhs.process_id_to_device_ids()) {
    return false;
  }
  return true;
}

}  // namespace phi::distributed::auto_parallel
