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

#include "paddle/phi/common/place.h"

#include <sstream>
#include <string>

#include "glog/logging.h"
#include "paddle/phi/api/ext/exception.h"
#include "paddle/phi/backends/gpu/gpu_info.h"

namespace phi {

const char *AllocationTypeStr(AllocationType type) {
  switch (type) {
    case AllocationType::UNDEFINED:
      return "undefined";
    case AllocationType::CPU:
      return "cpu";
    case AllocationType::GPU:
      return "gpu";
    case AllocationType::GPUPINNED:
      return "gpu_pinned";
    case AllocationType::XPU:
      return "xpu";
    case AllocationType::IPU:
      return "ipu";
    default:
      PD_THROW("Invalid phi device type.");
      return {};
  }
}

Place::Place(AllocationType type, const std::string &dev_type)
    : device(0),
      alloc_type_(type),
      device_type_id_(phi::CustomRegisteredDeviceMap::Instance()
                          .GetOrRegisterGlobalDeviceTypeId(dev_type)) {}

std::string Place::DebugString() const {
  std::ostringstream os;
  os << "Place(";
  if (alloc_type_ == AllocationType::CUSTOM) {
    os << phi::CustomRegisteredDeviceMap::Instance().GetGlobalDeviceType(
        device_type_id_);
  } else {
    os << AllocationTypeStr(alloc_type_);
  }
  if (alloc_type_ == AllocationType::GPUPINNED ||
      alloc_type_ == AllocationType::CPU) {
    os << ")";
  } else {
    os << ":" << std::to_string(device) << ")";
  }
  return os.str();
}

std::ostream &operator<<(std::ostream &os, const Place &p) {
  os << p.DebugString();
  return os;
}

Place GetPinnedPlace(const Place &place) {
  switch (place.GetType()) {
    case AllocationType::GPU:
      return phi::GPUPinnedPlace();
      break;
    default:
      return place;
  }
}

CustomRegisteredDeviceMap &CustomRegisteredDeviceMap::Instance() {
  static CustomRegisteredDeviceMap g_custom_registered_device_map;
  return g_custom_registered_device_map;
}

size_t CustomRegisteredDeviceMap::GetOrRegisterGlobalDeviceTypeId(
    const std::string &device_type) {
  if (device_type.empty()) return 0;
  if (registered_device_type_id_.find(device_type) ==
      registered_device_type_id_.end()) {
    size_t device_type_id = registered_device_type_id_.size() + 1;
    registered_device_type_id_[device_type] = device_type_id;
    registered_device_type_[device_type_id] = device_type;
  }
  return registered_device_type_id_[device_type];
}

std::string CustomRegisteredDeviceMap::GetGlobalDeviceType(
    size_t device_type_id) {
  if (registered_device_type_.find(device_type_id) ==
      registered_device_type_.end())
    return "";
  return registered_device_type_[device_type_id];
}

constexpr static int kAllocationTypeBitLength = 8;
constexpr static int kDeviceTypeIDBitLength = 8;
constexpr static int kDeviceIDBitLength = 8;

uint32_t Place::Hash::operator()(const Place &place) const {
  uint32_t hash_value = 0;
  // |----31-24------|-----23-16------|-----15-08----|---7-0----|
  // | For extension | AllocationType | DeviceTypeID | DeviceID |
  hash_value |= (static_cast<uint8_t>(place.alloc_type_)
                 << (kDeviceIDBitLength + kDeviceTypeIDBitLength));
  hash_value |=
      (static_cast<uint8_t>(place.device_type_id_) << kDeviceIDBitLength);
  hash_value |= static_cast<uint8_t>(place.device);
  return hash_value;
}

namespace detail {
static int8_t GetCorrectDeviceIdByPlaceType(
    const paddle::PlaceType &place_type) {
  switch (place_type) {
    case paddle::PlaceType::kCPU:
      return 0;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    case paddle::PlaceType::kGPU:
      return phi::backends::gpu::GetCurrentDeviceId();
#endif
    default:
      PD_THROW(
          "The PlaceType is a legacy design, only supports CPU and GPU, "
          "and will not support other place types in the future.");
  }
}
}  // namespace detail

Place::Place(paddle::PlaceType type)
    : device(detail::GetCorrectDeviceIdByPlaceType(type)),
      alloc_type_(static_cast<AllocationType>(type)),
      device_type_id_(
          CustomRegisteredDeviceMap::Instance().GetOrRegisterGlobalDeviceTypeId(
              "")) {
  LOG_FIRST_N(WARNING, 1)
      << "The `paddle::PlaceType::kCPU/kGPU` is deprecated since version "
         "2.3, and will be removed in version 2.4! Please use "
         "`paddle::CPUPlace()/DefaultGPUPlace()` to represent the place type.";
}

}  // namespace phi

namespace paddle {

bool operator==(const Place &place, PlaceType place_type) {
  LOG_FIRST_N(WARNING, 1)
      << "The `paddle::PlaceType::kCPU/kGPU` is deprecated since version "
         "2.3, and will be removed in version 2.4! Please use "
         "`Tensor::is_cpu()/is_gpu()` method to determine the type of place.";
  return place.GetType() == static_cast<AllocationType>(place_type);
}

bool operator==(PlaceType place_type, const Place &place) {
  LOG_FIRST_N(WARNING, 1)
      << "The `paddle::PlaceType::kCPU/kGPU` is deprecated since version "
         "2.3, and will be removed in version 2.4! Please use "
         "`Tensor::is_cpu()/is_gpu()` method to determine the type of place.";
  return static_cast<AllocationType>(place_type) == place.GetType();
}

GPUPlace DefaultGPUPlace() {
  return GPUPlace(
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::backends::gpu::GetCurrentDeviceId());
#else
      0);
#endif
}

}  // namespace paddle
