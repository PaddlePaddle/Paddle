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
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

#include "paddle/phi/core/distributed/auto_parallel/auto_parallel.pb.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace distributed {
namespace auto_parallel {
struct DeviceCapability {
  double single_precision_flops = 0.0;
  double double_precision_flops = 0.0;
  double memory_size_in_bytes = 0.0;
  double clock_rate_in_ghz = 0.0;

  // DeviceCapability from_string(const std::string& str);
  std::string to_string() const;

  static DeviceCapability from_proto(const DeviceCapabilityProto& proto);
  DeviceCapabilityProto to_proto() const;
};

inline std::ostream& operator<<(std::ostream& os, const DeviceCapability& obj) {
  os << obj.to_string();
  return os;
}

class Device {
 public:
  Device() = default;
  Device(int64_t global_id,
         int64_t local_id,
         int64_t machine_id,
         const std::string& type)
      : global_id_(global_id),
        local_id_(local_id),
        machine_id_(machine_id),
        type_(type) {}

  int64_t global_id() const { return global_id_; }
  int64_t local_id() const { return local_id_; }
  int64_t machine_id() const { return machine_id_; }
  const std::string& type() const { return type_; }

  const DeviceCapability& capability() const { return capability_; }
  void set_capability(const DeviceCapability& capability) {
    capability_ = capability;
  }

  // Device from_string(const std::string& mesh_str);
  std::string to_string() const;

  static Device from_proto(const DeviceProto& proto);
  DeviceProto to_proto() const;

 private:
  int64_t global_id_;
  int64_t local_id_;
  int64_t machine_id_;
  std::string type_;
  DeviceCapability capability_;
};

inline std::ostream& operator<<(std::ostream& os, const Device& obj) {
  os << obj.to_string();
  return os;
}

bool operator==(const Device& lhs, const Device& rhs);

inline bool operator!=(const Device& lhs, const Device& rhs) {
  return !operator==(lhs, rhs);
}

struct LinkCapability {
  double bandwidth = 0.0;  // Bytes/s
  double latency = 0.0;

  // LinkCapability from_string(const std::string& str);
  std::string to_string() const;

  static LinkCapability from_proto(const LinkCapabilityProto& proto);
  LinkCapabilityProto to_proto() const;
};

inline std::ostream& operator<<(std::ostream& os, const LinkCapability& obj) {
  os << obj.to_string();
  return os;
}

class Link {
 public:
  Link() = default;

  Link(int64_t source_id, int64_t target_id, const std::string& type)
      : source_id_(source_id), target_id_(target_id), type_(type) {}

  int64_t source_id() const { return source_id_; }
  int64_t target_id() const { return target_id_; }
  const std::string& type() const { return type_; }

  const LinkCapability& capability() const { return capability_; }
  void set_capability(const LinkCapability& capability) {
    capability_ = capability;
  }

  // Link from_string(const std::string& str);
  std::string to_string() const;

  static Link from_proto(const LinkProto& proto);
  LinkProto to_proto() const;

 private:
  int64_t source_id_;
  int64_t target_id_;
  std::string type_;
  LinkCapability capability_;
};

inline std::ostream& operator<<(std::ostream& os, const Link& obj) {
  os << obj.to_string();
  return os;
}

bool operator==(const Link& lhs, const Link& rhs);

inline bool operator!=(const Link& lhs, const Link& rhs) {
  return !operator==(lhs, rhs);
}

class Machine {
 public:
  Machine() = default;

  explicit Machine(int64_t id) : id_(id) {}

  int64_t id() const { return id_; }

  void set_id(int64_t id) { id_ = id; }

  const std::unordered_map<int64_t, const Device*>& devices() const {
    return devices_;
  }

  const std::unordered_map<int64_t, std::unordered_map<int64_t, const Link*>>&
  links() const {
    return links_;
  }

  const Device& device(int64_t global_id) const {
    return *devices_.at(global_id);
  }

  const Link& link(int64_t source_id, int64_t target_id) const {
    return *links_.at(source_id).at(target_id);
  }

  bool contains(int64_t device_id) const;

  void add_device(const Device& device);

  void add_link(const Link& link);

  // Machine from_string(const std::string& str);
  std::string to_string() const;

 private:
  int64_t id_ = -1;
  std::unordered_map<int64_t, const Device*> devices_;
  std::unordered_map<int64_t, std::unordered_map<int64_t, const Link*>> links_;
};

class DeviceMesh {
 public:
  DeviceMesh() = default;

  DeviceMesh(const std::string& name,
             const std::vector<int64_t>& shape,
             const std::vector<int64_t>& device_ids,
             const std::vector<std::string>& dim_names);

  const std::string& name() const { return name_; }

  void set_name(const std::string& name) { name_ = name; }

  const std::vector<int64_t>& shape() const { return shape_; }

  const std::vector<int64_t>& device_ids() const { return device_ids_; }

  const std::vector<std::string>& dim_names() const { return dim_names_; }

  std::string device_type() const {
    if (empty()) return "UNKNOWN";
    if (devices_.empty())
      return "UNKNOWN";
    else
      return std::begin(devices_)->second.type();
  }

  const std::unordered_map<int64_t, Device>& devices() const {
    return devices_;
  }

  const std::unordered_map<int64_t, std::unordered_map<int64_t, Link>>& links()
      const {
    return links_;
  }

  const std::unordered_map<int64_t, Machine>& machines() const {
    return machines_;
  }

  const Device& device(int64_t global_id) const {
    return devices_.at(global_id);
  }

  const Link& link(int64_t source_id, int64_t target_id) const {
    return links_.at(source_id).at(target_id);
  }

  const Machine& machine(int64_t machine_id) const {
    return machines_.at(machine_id);
  }

  int64_t size() const;
  int64_t ndim() const { return shape_.size(); }

  int64_t dim_size(int64_t dim) const {
    int64_t cdim = canonical_dim(dim, shape_.size());
    return shape_[cdim];
  }

  int64_t dim_size(const std::string& dim_name) const {
    for (std::size_t i = 0; i < dim_names_.size(); ++i) {
      if (dim_names_[i] == dim_name) {
        return shape_[i];
      }
    }
    PADDLE_THROW(errors::InvalidArgument(
        "Cannot find the dimension of %s in this device mesh.", dim_name));
  }

  bool empty() const { return (shape_.empty() || device_ids_.empty()); }
  bool contains(int64_t device_id) const;

  void add_device(const Device& device);
  void add_link(const Link& link);

  // DeviceMesh from_string(const std::string& mesh_str);
  std::string to_string() const;

  static DeviceMesh from_proto(const DeviceMeshProto& proto);
  DeviceMeshProto to_proto() const;

 private:
  std::string name_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> device_ids_;
  std::vector<std::string> dim_names_;
  std::unordered_map<int64_t, Device> devices_;
  std::unordered_map<int64_t, std::unordered_map<int64_t, Link>> links_;
  std::unordered_map<int64_t, Machine> machines_;
};

inline std::ostream& operator<<(std::ostream& os, const DeviceMesh& obj) {
  os << obj.to_string();
  return os;
}

bool operator==(const DeviceMesh& lhs, const DeviceMesh& rhs);

inline bool operator!=(const DeviceMesh& lhs, const DeviceMesh& rhs) {
  return !operator==(lhs, rhs);
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace phi
