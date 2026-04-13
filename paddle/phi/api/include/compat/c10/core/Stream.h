// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>

namespace c10 {

using StreamId = int64_t;

struct StreamData3 {
  StreamId stream_id;
  DeviceIndex device_index;
  DeviceType device_type;
};

class Stream final {
 private:
  Device device_;
  StreamId id_;

 public:
  enum Unsafe { UNSAFE };
  enum Default { DEFAULT };

  explicit Stream(Unsafe /*unused*/, Device device, StreamId id)
      : device_(device), id_(id) {}

  explicit Stream(Default /*unused*/, Device device)
      : device_(device), id_(0) {}

  bool operator==(const Stream& other) const noexcept {
    return this->device_ == other.device_ && this->id_ == other.id_;
  }
  bool operator!=(const Stream& other) const noexcept {
    return !(*this == other);
  }

  Device device() const noexcept { return device_; }
  DeviceType device_type() const noexcept { return device_.type(); }
  DeviceIndex device_index() const noexcept { return device_.index(); }
  StreamId id() const noexcept { return id_; }

  void* native_handle() const;

  template <typename T>
  void wait(const T& event) const {
    event.block(*this);
  }

  bool query() const;

  void synchronize() const;

  uint64_t hash() const noexcept {
    uint64_t bits = static_cast<uint64_t>(device_type()) << 56 |
                    static_cast<uint64_t>(device_index()) << 48 |
                    (static_cast<uint64_t>(id()) & ((1ull << 48) - 1));
    return bits;
  }

  struct StreamData3 pack3() const {
    return {id(), device_index(), device_type()};
  }

  static Stream unpack3(StreamId stream_id,
                        DeviceIndex device_index,
                        DeviceType device_type) {
    PD_CHECK(isValidDeviceType(device_type));
    return Stream(UNSAFE, Device(device_type, device_index), stream_id);
  }
};

inline std::ostream& operator<<(std::ostream& os, const Stream& s) {
  // Format: "stream {id} on device {device_type}:{device_index}"
  os << "stream " << s.id() << " on device " << s.device();
  return os;
}

}  // namespace c10

namespace std {
template <>
struct hash<c10::Stream> {
  size_t operator()(const c10::Stream& s) const noexcept {
    return std::hash<uint64_t>{}(s.hash());
  }
};
}  // namespace std
