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

namespace pten {

enum class AllocationType : int8_t {
  UNDEF = 0,
  CPU = 1,
  GPU = 2,
  GPUPINNED = 3,
  XPU = 4,
  NPU = 5,
  NPUPINNED = 6,
  IPU = 7,
  MLU = 8,
};

const char *AllocationTypeStr(AllocationType type);

/// \brief The place is used to specify where the data is stored.
class Place {
 public:
  Place() : device(0), alloc_type_(AllocationType::UNDEF) {}

  explicit Place(AllocationType type, int8_t id)
      : device(id), alloc_type_(type) {}

  explicit Place(AllocationType type) : device(0), alloc_type_(type) {}

  void Reset(AllocationType type, int8_t device_id = 0) noexcept {
    alloc_type_ = type;
    device = device_id;
  }

  AllocationType GetType() const { return alloc_type_; }

  int8_t GetDeviceId() const { return device; }

  std::string DebugString() const;

 public:
  // TODO(wilber): Just because of backward compatibility, it needs to be
  // changed to private in the future.
  int8_t device;

 private:
  AllocationType alloc_type_;
};

class CPUPlace : public Place {
 public:
  CPUPlace() : Place(AllocationType::CPU, 0) {}
};

class GPUPlace : public Place {
 public:
  GPUPlace() : Place(AllocationType::GPU, 0) {}
  explicit GPUPlace(int device_id) : Place(AllocationType::GPU, device_id) {}
};

class GPUPinnedPlace : public Place {
 public:
  GPUPinnedPlace() : Place(AllocationType::GPUPINNED) {}
};

class XPUPlace : public Place {
 public:
  XPUPlace() : Place(AllocationType::XPU, 0) {}
  explicit XPUPlace(int device_id) : Place(AllocationType::XPU, device_id) {}
};

class NPUPlace : public Place {
 public:
  NPUPlace() : Place(AllocationType::NPU, 0) {}
  explicit NPUPlace(int device_id) : Place(AllocationType::XPU, device_id) {}
};

class NPUPinnedPlace : public Place {
 public:
  NPUPinnedPlace() : Place(AllocationType::NPUPINNED) {}
};

class IPUPlace : public Place {
 public:
  IPUPlace() : Place(AllocationType::XPU, 0) {}
  explicit IPUPlace(int device_id) : Place(AllocationType::XPU, device_id) {}
};

class MLUPlace : public Place {
 public:
  MLUPlace() : Place(AllocationType::MLU, 0) {}
  explicit MLUPlace(int device_id) : Place(AllocationType::MLU, device_id) {}
};

std::ostream &operator<<(std::ostream &, const Place &);

}  // namespace pten
