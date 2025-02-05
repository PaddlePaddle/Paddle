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
#include <iterator>

#include "paddle/phi/core/distributed/auto_parallel/device_mesh.h"
#include "paddle/phi/core/distributed/auto_parallel/proto_helper.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
namespace phi::distributed::auto_parallel {

std::string DeviceCapability::to_string() const {
  std::string str;
  str += "{sflops: " + to_string_with_precision(single_precision_flops) + ", ";
  str += "dflops: " + to_string_with_precision(double_precision_flops) + ", ";
  str += "memory: " + to_string_with_precision(memory_size_in_bytes) + ", ";
  str += "rate: " + to_string_with_precision(clock_rate_in_ghz) + "}";
  return str;
}

DeviceCapability DeviceCapability::from_proto(
    const DeviceCapabilityProto &proto) {
  DeviceCapability capability;
  capability.single_precision_flops = proto.single_precision_flops();
  capability.double_precision_flops = proto.double_precision_flops();
  capability.memory_size_in_bytes = proto.memory_size_in_bytes();
  capability.clock_rate_in_ghz = proto.clock_rate_in_ghz();
  return capability;
}

void DeviceCapability::to_proto(DeviceCapabilityProto *proto) const {
  proto->set_single_precision_flops(single_precision_flops);
  proto->set_double_precision_flops(double_precision_flops);
  proto->set_memory_size_in_bytes(memory_size_in_bytes);
  proto->set_clock_rate_in_ghz(clock_rate_in_ghz);
}

std::string Device::to_string() const {
  std::string str = "{global_id: " + std::to_string(global_id_) + ", ";
  str += "local_id: " + std::to_string(local_id_) + ", ";
  str += "machine_id: " + std::to_string(machine_id_) + ", ";
  str += "type: " + type_ + ", ";
  str += "capability: " + capability_.to_string() + "}";
  return str;
}

Device Device::from_proto(const DeviceProto &proto) {
  Device device;
  device.global_id_ = proto.global_id();
  device.local_id_ = proto.local_id();
  device.machine_id_ = proto.machine_id();
  device.type_ = proto.type();
  device.capability_ = DeviceCapability::from_proto(proto.capability());
  return device;
}

void Device::to_proto(DeviceProto *proto) const {
  proto->set_global_id(global_id_);
  proto->set_local_id(local_id_);
  proto->set_machine_id(machine_id_);
  proto->set_type(type_);
  proto->mutable_capability()->CopyFrom(
      phi::distributed::to_proto(capability_));
}

bool operator==(const Device &lhs, const Device &rhs) {
  if (lhs.global_id() != rhs.global_id()) {
    return false;
  }
  if (lhs.local_id() != rhs.local_id()) {
    return false;
  }
  if (lhs.machine_id() != rhs.machine_id()) {
    return false;
  }
  if (lhs.type() != rhs.type()) {
    return false;
  }
  return true;
}

std::string LinkCapability::to_string() const {
  std::string str;
  str += "{bandwidth: " + to_string_with_precision(bandwidth) + ",";
  str += "latency: " + to_string_with_precision(latency) + "}";
  return str;
}

LinkCapability LinkCapability::from_proto(const LinkCapabilityProto &proto) {
  LinkCapability capability;
  capability.bandwidth = proto.bandwidth();
  capability.latency = proto.latency();
  return capability;
}

void LinkCapability::to_proto(LinkCapabilityProto *proto) const {
  proto->set_bandwidth(bandwidth);
  proto->set_latency(latency);
}

std::string Link::to_string() const {
  std::string str = "{source_id:" + std::to_string(source_id_) + ",";
  str += "target_id:" + std::to_string(target_id_) + ",";
  str += "type:" + type_ + ",";
  str += "capability:" + capability_.to_string() + "}";
  return str;
}

Link Link::from_proto(const LinkProto &proto) {
  Link link;
  link.source_id_ = proto.source_id();
  link.target_id_ = proto.target_id();
  link.type_ = proto.type();
  link.capability_ = LinkCapability::from_proto(proto.capability());
  return link;
}

void Link::to_proto(LinkProto *proto) const {
  proto->set_source_id(source_id_);
  proto->set_target_id(target_id_);
  proto->set_type(type_);
  proto->mutable_capability()->CopyFrom(
      phi::distributed::to_proto(capability_));
}

bool operator==(const Link &lhs, const Link &rhs) {
  if (lhs.source_id() != rhs.source_id()) {
    return false;
  }
  if (lhs.target_id() != rhs.target_id()) {
    return false;
  }
  if (lhs.type() != rhs.type()) {
    return false;
  }
  return true;
}

bool Machine::contains(int64_t device_id) const {
  if (devices_.count(device_id) == 1) {
    return true;
  } else {
    return false;
  }
}

void Machine::add_device(const Device &device) {
  if (id() == -1) {
    set_id(device.machine_id());
  } else {
    PADDLE_ENFORCE_EQ(device.machine_id(),
                      id(),
                      errors::InvalidArgument(
                          "The machine id [%d] of the device should be equal "
                          "to this machine id [%d].",
                          device.machine_id(),
                          id_));
  }
  devices_[device.global_id()] = &device;
}

void Machine::add_link(const Link &link) {
  PADDLE_ENFORCE_EQ(contains(link.source_id()),
                    true,
                    errors::InvalidArgument(
                        "The source device id of the added link [%s] "
                        "cannot be found in the device_ids. Please add the "
                        "source device before adding this link",
                        std::to_string(link.source_id())));
  links_[link.source_id()][link.target_id()] = &link;
}

std::string Machine::to_string() const {
  std::string str = "{devices: [";
  for (const auto &device : devices_) {
    str += device.second->to_string() + ", ";
  }
  str.replace(str.size() - 2, 2, "], ");

  str += "links: [";
  for (const auto &item : links_) {
    str += "{";
    str += "source_id: " + std::to_string(item.first) + ", neighbors: [";
    for (const auto &link : item.second) {
      str += link.second->to_string() + ", ";
    }
    str.replace(str.size() - 2, 2, "]}, ");
  }
  str.replace(str.size() - 4, 4, "]}");
  return str;
}

DeviceMesh::DeviceMesh(const std::string &name,
                       const std::vector<int64_t> &shape,
                       const std::vector<int64_t> &device_ids,
                       const std::vector<std::string> &dim_names) {
  name_ = name;
  shape_ = shape;
  int64_t size = this->size();

  PADDLE_ENFORCE_EQ(
      size,
      device_ids.size(),
      errors::InvalidArgument("The size %d of this device mesh must be "
                              "equal to the size %d of its device ids.",
                              size,
                              device_ids.size()));
  PADDLE_ENFORCE_EQ(
      has_duplicates(device_ids),
      false,
      errors::InvalidArgument("The device ids [%s] must be unique.",
                              str_join(device_ids)));
  device_ids_ = device_ids;

  PADDLE_ENFORCE_EQ(
      shape_.size(),
      dim_names.size(),
      errors::InvalidArgument(
          "The size %d of mesh shape must be equal to the size %d "
          "of the dimension names.",
          shape_.size(),
          dim_names.size()));
  PADDLE_ENFORCE_EQ(has_duplicates(dim_names),
                    false,
                    errors::InvalidArgument(
                        "The names [%s] of each dimension must be unique.",
                        str_join(dim_names)));
  dim_names_ = dim_names;
}

int64_t DeviceMesh::size() const {
  if (shape_.empty()) return 0;
  int64_t size = 1;
  for (const int64_t dim_size : shape_) size *= dim_size;
  return size;
}

bool DeviceMesh::contains(int64_t device_id) const {
  auto result =
      std::find(std::begin(device_ids_), std::end(device_ids_), device_id);
  if (result != std::end(device_ids_)) {
    return true;
  } else {
    return false;
  }
}

void DeviceMesh::add_device(const Device &device) {
  PADDLE_ENFORCE_EQ(
      contains(device.global_id()),
      true,
      errors::InvalidArgument(
          "The added device id [%s] cannot be found in the device_ids.",
          std::to_string(device.global_id())));
  // Operator [] will create a new object if it cannot find one.
  // So we add the default constructor for Device and Machine
  // to make sure the new object can be created.
  devices_[device.global_id()] = device;
  machines_[device.machine_id()].add_device(devices_[device.global_id()]);
}

void DeviceMesh::add_link(const Link &link) {
  PADDLE_ENFORCE_EQ(
      contains(link.source_id()),
      true,
      errors::InvalidArgument("The source id of the added link [%s] "
                              "cannot be found in the device_ids.",
                              std::to_string(link.source_id())));
  PADDLE_ENFORCE_EQ(
      contains(link.target_id()),
      true,
      errors::InvalidArgument("The source id of the added link [%s] "
                              "cannot be found in the device_ids.",
                              std::to_string(link.target_id())));
  // Operator [] will create a new object if it cannot find one.
  // So we add the default constructor for Device and Machine
  // to make sure the new object can be created.
  links_[link.source_id()][link.target_id()] = link;
  const Device &source_device = devices_[link.source_id()];
  machines_[source_device.machine_id()].add_link(
      links_[link.source_id()][link.target_id()]);
}

std::string DeviceMesh::to_string() const {
  std::string mesh_str = "{name: " + name_ + ", ";
  mesh_str += "shape: [" + str_join(shape_) + "], ";
  mesh_str += "device_ids: [" + str_join(device_ids_) + "], ";
  mesh_str += "dim_names: [" + str_join(dim_names_) + "], ";
  mesh_str += "\ndevices: [\n";
  for (const auto &device : devices_) {
    mesh_str += "  " + device.second.to_string() + ",\n";
  }
  mesh_str.replace(mesh_str.size() - 2, 2, "],");

  mesh_str += "\nlinks: [\n";
  for (const auto &item : links_) {
    mesh_str += "  {";
    mesh_str += "source_id: " + std::to_string(item.first) + ", neighbors: [";
    for (const auto &link : item.second) {
      mesh_str += link.second.to_string() + ", ";
    }
    mesh_str.replace(mesh_str.size() - 2, 2, "]},\n");
  }
  mesh_str.replace(mesh_str.size() - 4, 4, "]}");
  return mesh_str;
}

DeviceMesh DeviceMesh::from_proto(const DeviceMeshProto &proto) {
  DeviceMesh mesh;

  mesh.name_ = proto.name();

  mesh.shape_.resize(proto.shape_size());
  for (int i = 0; i < proto.shape_size(); ++i) {
    mesh.shape_[i] = proto.shape(i);
  }

  mesh.device_ids_.resize(proto.device_ids_size());
  for (int i = 0; i < proto.device_ids_size(); ++i) {
    mesh.device_ids_[i] = proto.device_ids(i);
  }

  mesh.dim_names_.resize(proto.dim_names_size());
  for (int i = 0; i < proto.dim_names_size(); ++i) {
    mesh.dim_names_[i] = proto.dim_names(i);
  }

  for (int i = 0; i < proto.devices_size(); ++i) {
    mesh.add_device(Device::from_proto(proto.devices(i)));
  }

  for (int i = 0; i < proto.links_size(); ++i) {
    mesh.add_link(Link::from_proto(proto.links(i)));
  }

  return mesh;
}

void DeviceMesh::to_proto(DeviceMeshProto *proto) const {
  proto->set_name(name_);

  for (const auto &i : shape_) {
    proto->add_shape(i);
  }

  for (const auto &i : device_ids_) {
    proto->add_device_ids(i);
  }

  for (const auto &i : dim_names_) {
    proto->add_dim_names(i);
  }

  for (const auto &device : devices_) {
    proto->mutable_devices()->Add()->CopyFrom(
        phi::distributed::to_proto(device.second));
  }

  for (const auto &neighbors : links_) {
    for (const auto &link : neighbors.second) {
      proto->mutable_links()->Add()->CopyFrom(
          phi::distributed::to_proto(link.second));
    }
  }
}

bool operator==(const DeviceMesh &lhs, const DeviceMesh &rhs) {
  // Use the unique name to do the fast comparison
  if (lhs.name() != rhs.name()) {
    return false;
  }
  return true;
}

}  // namespace phi::distributed::auto_parallel
