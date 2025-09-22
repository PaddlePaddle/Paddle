// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include <c10/core/DeviceType.h>

namespace c10 {
using DeviceIndex = int8_t;

struct Device final {
  using Type = DeviceType;
  Device(phi::Place place) : inner_(place) {}
  Device(DeviceType type, DeviceIndex index = 0)
      : inner_(phi::Place(type, index)) {}  // NOLINT

  DeviceIndex index() const noexcept { return inner_.GetDeviceId(); }

  DeviceType type() const { return inner_.GetType(); }

  phi::Place _PD_GetInner() const { return inner_; }

 private:
  phi::Place inner_;
};

}  // namespace c10

namespace at {
using c10::Device;
using c10::DeviceIndex;
}  // namespace at

namespace torch {
using c10::Device;
using c10::DeviceIndex;
}  // namespace torch
