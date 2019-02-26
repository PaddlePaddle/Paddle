// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <glog/logging.h>
#include "target_wrapper.h"

namespace paddle {
namespace lite {

static void* TargetMalloc(TargetType target, size_t size) {
  void* data{nullptr};
  switch (static_cast<int>(target)) {
    case static_cast<int>(TargetType::kX86):
      data = TargetWrapper<TARGET(kX86)>::Malloc(size);
      break;
    case static_cast<int>(TargetType::kCUDA):
      data = TargetWrapper<TARGET(kCUDA)>::Malloc(size);
      break;
    case static_cast<int>(TargetType::kARM):
      data = TargetWrapper<TARGET(kARM)>::Malloc(size);
      break;
    case static_cast<int>(TargetType::kHost):
      data = TargetWrapper<TARGET(kHost)>::Malloc(size);
      break;
    default:
      LOG(FATAL) << "Unknown type";
  }
  return data;
}

static void TargetFree(TargetType target, void* data) {
  switch (static_cast<int>(target)) {
    case static_cast<int>(TargetType::kX86):
      TargetWrapper<TARGET(kX86)>::Free(data);
      break;
    case static_cast<int>(TargetType::kCUDA):
      TargetWrapper<TARGET(kX86)>::Free(data);
      break;
    case static_cast<int>(TargetType::kARM):
      TargetWrapper<TARGET(kX86)>::Free(data);
      break;
    default:
      LOG(FATAL) << "Unknown type";
  }
}

// Memory buffer manager.
class Buffer {
 public:
  Buffer() = default;
  Buffer(TargetType target, size_t size) : space_(size), target_(target) {}

  void* data() const { return data_; }

  void ResetLazy(TargetType target, size_t size) {
    if (target != target_ || space_ < size) {
      Free();
    }

    if (size < space_) return;
    data_ = TargetMalloc(target, size);
    target_ = target;
    space_ = size;
  }

  void ResizeLazy(size_t size) { ResetLazy(target_, size); }

  void Free() {
    if (space_ > 0) {
      TargetFree(target_, data_);
    }
    target_ = TargetType::kHost;
    space_ = 0;
  }

 private:
  size_t space_{0};
  void* data_{nullptr};
  TargetType target_{TargetType::kHost};
};

}  // namespace lite
}  // namespace paddle
