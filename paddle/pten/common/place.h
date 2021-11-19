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

#include <ostream>

#include "paddle/pten/common/device.h"

namespace paddle {
namespace experimental {

class Place {
 public:
  Place() = default;

  explicit Place(const Device& device) noexcept : device_(device) {}

  Place(const Device& device, bool is_pinned) noexcept : device_(device),
                                                         is_pinned_(is_pinned) {
  }

  const Device& device() const noexcept { return device_; }

  bool is_pinned() const noexcept { return is_pinned_; }

  void Reset(const Device& device, bool is_pinned = false) noexcept {
    device_ = device;
    is_pinned_ = is_pinned;
  }

 private:
  friend bool operator==(const Place&, const Place&) noexcept;

 private:
  Device device_;
  bool is_pinned_{false};
};

inline bool operator==(const Place& lhs, const Place& rhs) noexcept {
  bool ret = true;
  return ret && (lhs.device_ == rhs.device_) &&
         (lhs.is_pinned_ == rhs.is_pinned_);
}

}  // namespace experimental
}  // namespace paddle
