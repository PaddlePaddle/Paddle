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

#include <string>
#include <unordered_map>

#include "paddle/phi/api/include/dll_decl.h"
#include "paddle/phi/core/macros.h"
namespace paddle {
enum class PlaceType;
}

namespace phi {

enum class AllocationType : int8_t {
  UNDEFINED = 0,
  CPU = 1,
  GPU = 2,
  GPUPINNED = 3,
  XPU = 4,
  IPU = 7,
  CUSTOM = 9,
};

class CustomRegisteredDeviceMap {
 public:
  static CustomRegisteredDeviceMap& Instance();

  size_t GetOrRegisterGlobalDeviceTypeId(const std::string& device_type);

  std::string GetGlobalDeviceType(size_t device_type_id_);

 private:
  CustomRegisteredDeviceMap() = default;
  std::unordered_map<std::string, size_t> registered_device_type_id_;
  std::unordered_map<size_t, std::string> registered_device_type_;
};

const char* AllocationTypeStr(AllocationType type);

/// \brief The place is used to specify where the data is stored.
class PADDLE_API Place {
 public:
  Place() : device(0), alloc_type_(AllocationType::UNDEFINED) {}

  explicit Place(AllocationType type,
                 int8_t id,
                 const std::string& dev_type = "")
      : device(id),
        alloc_type_(type),
        device_type_id_(phi::CustomRegisteredDeviceMap::Instance()
                            .GetOrRegisterGlobalDeviceTypeId(dev_type)) {}

  explicit Place(AllocationType type, const std::string& dev_type = "")
      : device(0),
        alloc_type_(type),
        device_type_id_(phi::CustomRegisteredDeviceMap::Instance()
                            .GetOrRegisterGlobalDeviceTypeId(dev_type)) {}

  // See NOTE [ Why need to temporarily adapt to PlaceType? ]
  Place(paddle::PlaceType type);  // NOLINT

  void Reset(AllocationType type,
             int8_t device_id = 0,
             const std::string& dev_type = "") noexcept {
    alloc_type_ = type;
    device = device_id;
    if (!dev_type.empty()) {
      device_type_id_ = phi::CustomRegisteredDeviceMap::Instance()
                            .GetOrRegisterGlobalDeviceTypeId(dev_type);
    }
  }

  AllocationType GetType() const { return alloc_type_; }

  int8_t GetDeviceId() const { return device; }

  std::string GetDeviceType() const {
    return phi::CustomRegisteredDeviceMap::Instance().GetGlobalDeviceType(
        device_type_id_);
  }

  std::string DebugString() const;

  struct Hash {
    // Note: Now the number of bits we need does not exceed 32 bits, so there is
    // no need to use 64 bits. If needed in the future, it can be expanded,
    // but now we don’t over-design.
    uint32_t operator()(const Place& place) const;
  };

  uint32_t HashValue() const { return Hash()(*this); }

  inline bool operator==(const Place& rhs) const {
    return HashValue() == rhs.HashValue();
  }
  inline bool operator!=(const Place& rhs) const {
    return HashValue() != rhs.HashValue();
  }
  inline bool operator<(const Place& rhs) const {
    return HashValue() < rhs.HashValue();
  }

 public:
  // TODO(wilber): Just because of backward compatibility, it needs to be
  // changed to private in the future.
  int8_t device{0};

 private:
  AllocationType alloc_type_{AllocationType::UNDEFINED};
  size_t device_type_id_;
};

class CPUPlace : public Place {
 public:
  CPUPlace() : Place(AllocationType::CPU) {}

  CPUPlace(const CPUPlace&) = default;
  CPUPlace(const Place& place UNUSED) : Place(AllocationType::CPU) {}  // NOLINT
};

class GPUPlace : public Place {
 public:
  GPUPlace() : Place(AllocationType::GPU, 0) {}
  explicit GPUPlace(int device_id) : Place(AllocationType::GPU, device_id) {}

  GPUPlace(const GPUPlace&) = default;
  GPUPlace(const Place& place)  // NOLINT
      : Place(AllocationType::GPU, place.GetDeviceId()) {}
};

class GPUPinnedPlace : public Place {
 public:
  GPUPinnedPlace() : Place(AllocationType::GPUPINNED) {}

  GPUPinnedPlace(const GPUPinnedPlace&) = default;
  GPUPinnedPlace(const Place& place UNUSED)  // NOLINT
      : Place(AllocationType::GPUPINNED) {}
};

class XPUPlace : public Place {
 public:
  XPUPlace() : Place(AllocationType::XPU, 0) {}
  explicit XPUPlace(int device_id) : Place(AllocationType::XPU, device_id) {}

  XPUPlace(const XPUPlace&) = default;
  XPUPlace(const Place& place)  // NOLINT
      : Place(AllocationType::XPU, place.GetDeviceId()) {}
};

class IPUPlace : public Place {
 public:
  IPUPlace() : Place(AllocationType::IPU, 0) {}
  explicit IPUPlace(int device_id) : Place(AllocationType::IPU, device_id) {}

  IPUPlace(const IPUPlace&) = default;
  IPUPlace(const Place& place)  // NOLINT
      : Place(AllocationType::IPU, place.GetDeviceId()) {}
};

class CustomPlace : public Place {
 public:
  CustomPlace() : Place(AllocationType::CUSTOM, 0, "") {}
  explicit CustomPlace(const std::string dev_type)
      : Place(AllocationType::CUSTOM, 0, dev_type) {}
  CustomPlace(const std::string dev_type, int device_id)
      : Place(AllocationType::CUSTOM, device_id, dev_type) {}

  CustomPlace(const CustomPlace&) = default;
  CustomPlace(const Place& place) {  // NOLINT
    if (place.GetType() == AllocationType::CUSTOM) {
      this->Reset(
          AllocationType::CUSTOM, place.GetDeviceId(), place.GetDeviceType());
    }
  }
};

std::ostream& operator<<(std::ostream&, const Place&);

Place GetPinnedPlace(const Place& place);

}  // namespace phi

namespace paddle {
namespace experimental {
using AllocationType = phi::AllocationType;
using GPUPinnedPlace = phi::GPUPinnedPlace;
using XPUPlace = phi::XPUPlace;
}  // namespace experimental

using AllocationType = phi::AllocationType;
using Place = phi::Place;
using CPUPlace = phi::CPUPlace;
using GPUPlace = phi::GPUPlace;

/* NOTE [ Why need to temporarily adapt to PlaceType? ]

`PlaceType` enum class is the place type used by custom operators since the
release of 2.0. Since 2.3, we have refactored the operator library and designed
a new external Place type. The original PlaceType is no longer suitable for use
as an internal type of the framework, but immediately delete the PlaceType,
it will cause the previous custom operators to be incompatible, so it cannot be
deleted in the short term. We'd better delete this abandoned data type in 2.4.

Note: This type cannot add any new type!!! It is only used for compatibility
with
historical writing and we will remove this temporary type in the future.
This Type cannot be used in framework! only used for custom operator!

The original PlaceType define:

- enum class PlaceType { kUNK = -1, kCPU, kGPU };

The historical PlaceType using:

- PD_CHECK(x.place() == paddle::PlaceType::kCPU)
- auto out = paddle::Tensor(paddle::PlaceType::kCPU, x.shape());

*/
enum class PlaceType {
  kUNK = static_cast<int>(phi::AllocationType::UNDEFINED),
  kCPU = static_cast<int>(phi::AllocationType::CPU),
  kGPU = static_cast<int>(phi::AllocationType::GPU),
};

PADDLE_API bool operator==(const Place& place, PlaceType place_type);
PADDLE_API bool operator==(PlaceType place_type, const Place& place);

PADDLE_API GPUPlace DefaultGPUPlace();

}  // namespace paddle
