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

#include <popart/devicemanager.hpp>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace ipu {

enum class DeviceType { IpuModel = 0, Cpu, Ipu, OfflineIpu, Sim };

class Device {
 public:
  Device() {}
  explicit Device(const popart::DeviceInfo& device_info);

  int getId() const { return id_; }
  bool isAttached() const { return is_attached_; }
  DeviceType getType() const { return device_type_; }

 private:
  int id_;
  bool is_attached_;
  DeviceType device_type_;
  /* TODO:: Add more elements in the future */
};

}  // namespace ipu
}  // namespace platform
}  // namespace paddle
