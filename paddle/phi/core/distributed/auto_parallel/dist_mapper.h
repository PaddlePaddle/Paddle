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
#pragma once

#include <utility>

#include "paddle/phi/core/distributed/auto_parallel/auto_parallel.pb.h"
#include "paddle/phi/core/distributed/auto_parallel/device_mesh.h"
#include "paddle/phi/core/distributed/auto_parallel/process_mesh.h"

namespace phi {
namespace distributed {
namespace auto_parallel {

class DistributedMapper {
 public:
  DistributedMapper() = default;

  const std::map<std::string, DeviceMesh>& device_meshes() const {
    return device_meshes_;
  }

  const DeviceMesh& device_mesh(const std::string& name) const {
    return device_meshes_.at(name);
  }

  void add_device_mesh(const DeviceMesh& device_mesh) {
    device_meshes_[device_mesh.name()] = device_mesh;
  }

  const std::map<int64_t, std::pair<std::string, std::vector<int64_t>>>&
  process_id_to_device_ids() const {
    return process_id_to_device_ids_;
  }

  void set_process_id_to_device_ids(
      const std::map<int64_t, std::pair<std::string, std::vector<int64_t>>>&
          process_id_to_device_ids);

  // DistributedMapper from_string(const std::string& mapper_str);
  std::string to_string() const;

  static DistributedMapper from_proto(const DistributedMapperProto& proto);
  DistributedMapperProto to_proto() const;

 private:
  std::map<std::string, DeviceMesh> device_meshes_;
  std::map<int64_t, std::pair<std::string, std::vector<int64_t>>>
      process_id_to_device_ids_;
};

bool operator==(const DistributedMapper& lhs, const DistributedMapper& rhs);

inline std::ostream& operator<<(std::ostream& os,
                                const DistributedMapper& obj) {
  os << obj.to_string();
  return os;
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace phi
