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

#include <algorithm>
#include <unordered_map>

#include "paddle/fluid/distributed/auto_parallel/utils.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

std::string Device::to_string() const {
  std::string mesh_str = "{global_id:" + std::to_string(global_id) + ",";
  mesh_str += "local_id:" + std::to_string(local_id) + ",";
  mesh_str += "machine_id:" + std::to_string(machine_id) + ",";
  mesh_str += "type:" + type + ",";
  mesh_str +=
      "f32_flops:" + std::to_string(capability.single_precision_flops) + ",";
  mesh_str +=
      "f64_flops:" + std::to_string(capability.double_precision_flops) + ",";
  mesh_str += "memory:" + std::to_string(capability.memory_size_in_bytes) + ",";
  mesh_str += "rate:" + std::to_string(capability.clock_rate_in_ghz) + "}";
  return mesh_str;
}

std::string Link::to_string() const {
  std::string mesh_str = "{source_id:" + std::to_string(source_id) + ",";
  mesh_str += "target_id:" + std::to_string(target_id) + ",";
  mesh_str += "type:" + type + ",";
  mesh_str += "bandwidth:" + std::to_string(capability.bandwidth) + ",";
  mesh_str += "latency:" + std::to_string(capability.latency) + "}";
  return mesh_str;
}

DeviceMesh::DeviceMesh(
    const std::vector<int64_t> &shape,
    const std::vector<int64_t> &device_ids,
    const std::vector<std::string> &dim_names,
    const std::string &device_type,
    const std::vector<Device> &devices,
    const std::unordered_map<std::int64_t, std::vector<Link>> &links) {
  shape_ = shape;
  int64_t size = this->size();

  PADDLE_ENFORCE_EQ(size,
                    device_ids.size(),
                    platform::errors::InvalidArgument(
                        "The size %d of this device mesh must be "
                        "equal to the size %d of its device ids.",
                        size,
                        device_ids.size()));
  PADDLE_ENFORCE_EQ(
      has_duplicates(device_ids),
      false,
      platform::errors::InvalidArgument("The device ids [%s] must be unique.",
                                        str_join(device_ids)));
  device_ids_ = device_ids;

  PADDLE_ENFORCE_EQ(
      shape_.size(),
      dim_names.size(),
      platform::errors::InvalidArgument(
          "The size %d of mesh shape must be equal to the size %d "
          "of the dimension names.",
          shape_.size(),
          dim_names.size()));
  PADDLE_ENFORCE_EQ(has_duplicates(dim_names),
                    false,
                    platform::errors::InvalidArgument(
                        "The names [%s] of each dimension must be unique.",
                        str_join(dim_names)));
  dim_names_ = dim_names;

  device_type_ = device_type;

  PADDLE_ENFORCE_EQ(device_ids_.size(),
                    devices.size(),
                    platform::errors::InvalidArgument(
                        "The size of device_ids must be equal to the size "
                        "of devices.",
                        device_ids_.size(),
                        devices.size()));
  std::vector<int64_t> tmp_global_device_ids;
  std::unordered_map<int64_t, std::vector<int64_t>> tmp_local_device_ids;
  for (const Device &device : devices) {
    PADDLE_ENFORCE_EQ(device.type,
                      device_type,
                      platform::errors::InvalidArgument(
                          "The device_type %s of each device must be equal to "
                          "the device_type %s of this mesh.",
                          device.type,
                          device_type));
    tmp_global_device_ids.push_back(device.global_id);
    tmp_local_device_ids[device.global_id].push_back(device.local_id);
  }
  std::vector<int64_t> sorted_global_device_ids = device_ids_;
  std::sort(sorted_global_device_ids.begin(), sorted_global_device_ids.end());
  std::sort(tmp_global_device_ids.begin(), tmp_global_device_ids.end());
  PADDLE_ENFORCE_EQ(
      sorted_global_device_ids,
      tmp_global_device_ids,
      platform::errors::InvalidArgument(
          "The global device ids [%s] of all devices must be equal to "
          "device_ids [%s].",
          str_join(sorted_global_device_ids),
          str_join(tmp_global_device_ids)));
  for (const auto &local_device_ids : tmp_local_device_ids) {
    PADDLE_ENFORCE_EQ(
        has_duplicates(local_device_ids.second),
        false,
        platform::errors::InvalidArgument(
            "The local device ids [%s] of machine %d must be unique.",
            str_join(local_device_ids.second),
            local_device_ids.first));
  }
  devices_ = devices;

  for (const auto &item : links) {
    int64_t source_id = item.first;
    for (const auto &link : item.second) {
      PADDLE_ENFORCE_EQ(
          source_id,
          link.source_id,
          platform::errors::InvalidArgument(
              "The device id %d does not match the source id %d of this link.",
              source_id,
              link.source_id));
      PADDLE_ENFORCE_EQ(std::binary_search(sorted_global_device_ids.begin(),
                                           sorted_global_device_ids.end(),
                                           link.source_id),
                        true,
                        platform::errors::InvalidArgument(
                            "Cannot find the source id %d in this device mesh.",
                            link.source_id));
      PADDLE_ENFORCE_EQ(std::binary_search(sorted_global_device_ids.begin(),
                                           sorted_global_device_ids.end(),
                                           link.target_id),
                        true,
                        platform::errors::InvalidArgument(
                            "Cannot find the target id %d in this device mesh.",
                            link.source_id));
    }
  }
  links_ = links;
}

int64_t DeviceMesh::size() const {
  if (shape_.empty()) return 0;
  int64_t size = 1;
  for (const int64_t dim_size : shape_) size *= dim_size;
  return size;
}

std::string DeviceMesh::to_string() const {
  std::string mesh_str = "{id:" + std::to_string(id_) + ",";
  mesh_str += "shape:[" + str_join(shape_) + "],";
  mesh_str += "process_ids:[" + str_join(device_ids_) + "],";
  mesh_str += "dim_names:[" + str_join(dim_names_) + "],";
  mesh_str += "device_type:" + device_type_ + ",";
  mesh_str += "devices:[";
  for (const auto &device : devices_) {
    mesh_str += device.to_string() + ",";
  }
  mesh_str.replace(mesh_str.size() - 1, 1, "]");

  mesh_str += "links:[";
  for (const auto &item : links_) {
    mesh_str += "{";
    mesh_str += "source_id:" + std::to_string(item.first) + ", neighbors:[";
    for (const auto &link : item.second) {
      mesh_str += link.to_string() + ",";
    }
    mesh_str.replace(mesh_str.size() - 1, 1, "]");
    mesh_str += "},";
  }
  mesh_str.replace(mesh_str.size() - 1, 1, "]");
  mesh_str += "}";
  return mesh_str;
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
