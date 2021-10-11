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

#include "paddle/fluid/platform/place.h"
#include "paddle/tcmpt/allocator.h"

#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace tcmpt {

class EigenGpuContext {
 public:
  explicit EigenGpuContext(const platform::Place& place) : place_(place) {}

  void SetAllocator(Allocator* allocator) { allocator_ = allocator; }

  Allocator* allocator() const noexcept { return allocator_; }

  const platform::Place& place() const noexcept { return place_; }

  void SetDevice(Eigen::GpuDevice* device) { device_ = device; }

  Eigen::GpuDevice* device() { return device_; }

 private:
  platform::Place place_;
  Allocator* allocator_{nullptr};
  Eigen::GpuDevice* device_{nullptr};
};

}  // namespace tcmpt
}  // namespace paddle
