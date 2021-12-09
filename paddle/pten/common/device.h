/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <cstdint>
#include <string>

namespace paddle {
namespace experimental {

enum class DeviceType : int8_t {
  kUndef = 0,
  kHost = 1,
  kXpu = 2,
  kCuda = 3,
  kHip = 4,
  kNpu = 5,
};

const char* DeviceTypeStr(DeviceType type);

/// \brief The device is used to store hardware information. It has not yet
/// stored information related to the math acceleration library.
struct Device final {
 public:
  Device() = default;

  Device(DeviceType type, int8_t id);

  Device(DeviceType type);

  DeviceType type() const noexcept { return type_; }

  /// \brief Returns the index of the device. Here, -1 is used to indicate an
  /// invalid value, and 0 to indicate a default value.
  /// \return The index of the device.
  int8_t id() const noexcept { return id_; }

  void set_type(DeviceType type) noexcept { type_ = type; }

  void set_id(int8_t id) noexcept { id_ = id; }

  std::string DebugString() const;

 private:
  friend bool operator==(const Device&, const Device&) noexcept;

 private:
  DeviceType type_{DeviceType::kUndef};
  int8_t id_{-1};
};

inline bool operator==(const Device& lhs, const Device& rhs) noexcept {
  return (lhs.type_ == rhs.type_) && (lhs.id_ == rhs.id_);
}

}  // namespace experimental
}  // namespace paddle
