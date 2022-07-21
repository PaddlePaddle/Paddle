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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "paddle/fluid/distributed/auto_parallel/auto_parallel.pb.h"
#include "paddle/fluid/distributed/auto_parallel/utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

struct DeviceCapability {
  double single_precision_flops = 0;
  double double_precision_flops = 0;
  double memory_size_in_bytes = 0;
  double clock_rate_in_ghz = 0.0;
};

struct Device {
  int64_t global_id;
  int64_t local_id;
  int64_t machine_id;
  std::string type;
  DeviceCapability capability;

  Device from_string(const std::string& mesh_str);
  std::string to_string() const;

  Device from_proto(const DeviceProto& proto);
  DeviceProto to_proto() const;
};

inline std::ostream& operator<<(std::ostream& os, const Device& obj) {
  os << obj.to_string();
  return os;
}

inline bool operator==(const Device& lhs, const Device& rhs) {
  if (lhs.global_id != rhs.global_id) {
    return false;
  }
  if (lhs.local_id != rhs.local_id) {
    return false;
  }
  if (lhs.machine_id != rhs.machine_id) {
    return false;
  }
  if (lhs.type != rhs.type) {
    return false;
  }
  return true;
}

inline bool operator!=(const Device& lhs, const Device& rhs) {
  return !operator==(lhs, rhs);
}

struct LinkCapability {
  double bandwidth = 0.0;  // Bytes/s
  double latency = 0.0;
};

struct Link {
  int64_t source_id;
  int64_t target_id;
  std::string type;
  LinkCapability capability;

  Link from_string(const std::string& mesh_str);
  std::string to_string() const;

  Link from_proto(const LinkProto& proto);
  LinkProto to_proto() const;
};

inline std::ostream& operator<<(std::ostream& os, const Link& obj) {
  os << obj.to_string();
  return os;
}

inline bool operator==(const Link& lhs, const Link& rhs) {
  if (lhs.source_id != rhs.source_id) {
    return false;
  }
  if (lhs.target_id != rhs.target_id) {
    return false;
  }
  if (lhs.type != rhs.type) {
    return false;
  }
  return true;
}

inline bool operator!=(const Link& lhs, const Link& rhs) {
  return !operator==(lhs, rhs);
}

class DeviceMesh {
 public:
  DeviceMesh() = default;

  DeviceMesh(const std::vector<int64_t>& shape,
             const std::vector<int64_t>& device_ids,
             const std::vector<std::string>& dim_names,
             const std::string& device_type,
             const std::vector<Device>& devices,
             const std::unordered_map<int64_t, std::vector<Link>>& links);

  int64_t id() const { return id_; }

  void set_id(int64_t id) { id_ = id; }

  const std::vector<int64_t>& shape() const { return shape_; }

  const std::vector<int64_t>& device_ids() const { return device_ids_; }

  const std::vector<std::string>& dim_names() const { return dim_names_; }

  const std::string device_type() const { return device_type_; }

  const std::vector<Device>& devices() const { return devices_; }

  const std::unordered_map<int64_t, std::vector<Link>>& links() const {
    return links_;
  }

  int64_t size() const;

  int64_t ndim() const { return shape_.size(); }

  int64_t dim_size(int64_t dim) const {
    int64_t cdim = canonical_dim(dim, shape_.size());
    return shape_[cdim];
  }

  int64_t dim_size(std::string dim_name) const {
    for (std::size_t i = 0; i < dim_names_.size(); ++i) {
      if (dim_names_[i] == dim_name) {
        return shape_[i];
      }
    }
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Cannot find the dimension of %s in this device mesh.", dim_name));
  }

  DeviceMesh from_string(const std::string& mesh_str);
  std::string to_string() const;

  DeviceMesh from_proto(const DeviceMeshProto& proto);
  DeviceMeshProto to_proto() const;

 private:
  static int64_t generate_id() {
    static std::atomic<int64_t> cur_id{0};
    return cur_id++;
  }

  int64_t id_ = generate_id();
  std::vector<int64_t> shape_;
  std::vector<int64_t> device_ids_;
  std::vector<std::string> dim_names_;
  std::string device_type_;
  std::vector<Device> devices_;
  std::unordered_map<int64_t, std::vector<Link>> links_;
};

inline std::ostream& operator<<(std::ostream& os, const DeviceMesh& obj) {
  os << obj.to_string();
  return os;
}

inline bool operator==(const DeviceMesh& lhs, const DeviceMesh& rhs) {
  if (lhs.id() != rhs.id()) {
    return false;
  }
  if (lhs.shape() != rhs.shape()) {
    return false;
  }
  if (lhs.device_ids() != rhs.device_ids()) {
    return false;
  }
  if (lhs.device_type() != rhs.device_type()) {
    return false;
  }
  if (lhs.devices() != rhs.devices()) {
    return false;
  }
  if (lhs.links() != rhs.links()) {
    return false;
  }
  return true;
}

inline bool operator!=(const DeviceMesh& lhs, const DeviceMesh& rhs) {
  return !operator==(lhs, rhs);
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
