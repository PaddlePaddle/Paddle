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
  switch (target) {
    case TargetType::kHost:
    case TargetType::kX86:
      data = TargetWrapper<TARGET(kHost)>::Malloc(size);
      break;
    case TargetType::kCUDA:
      data = TargetWrapper<TARGET(kCUDA)>::Malloc(size);
      break;
    default:
      LOG(FATAL) << "Unknown supported target " << TargetToStr(target);
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
    default:
      LOG(FATAL) << "Unknown type";
  }
}

static void TargetCopy(TargetType target, void* dst, const void* src,
                       size_t size) {
  switch (target) {
    case TargetType::kX86:
    case TargetType::kHost:
      TargetWrapper<TARGET(kHost)>::MemcpySync(dst, src, size,
                                               IoDirection::DtoD);
      break;

    case TargetType::kCUDA:
      TargetWrapper<TARGET(kCUDA)>::MemcpySync(dst, src, size,
                                               IoDirection::DtoD);
      break;
    default:
      LOG(FATAL) << "unsupported type";
  }
}

// Memory buffer manager.
class Buffer {
 public:
  Buffer() = default;
  Buffer(TargetType target, size_t size) : space_(size), target_(target) {}

  void* data() const { return data_; }
  TargetType target() const { return target_; }
  size_t space() const { return space_; }

  void ResetLazy(TargetType target, size_t size) {
    if (target != target_ || space_ < size) {
      Free();
      data_ = TargetMalloc(target, size);
      target_ = target;
      space_ = size;
    }
  }

  void ResizeLazy(size_t size) { ResetLazy(target_, size); }

  void Free() {
    if (space_ > 0) {
      TargetFree(target_, data_);
    }
    target_ = TargetType::kHost;
    space_ = 0;
  }

  void CopyDataFrom(const Buffer& other, size_t nbytes) {
    target_ = other.target_;
    ResizeLazy(nbytes);
    // TODO(Superjomn) support copy between different targets.
    TargetCopy(target_, data_, other.data_, nbytes);
  }

 private:
  // memory it actually malloced.
  size_t space_{0};
  void* data_{nullptr};
  TargetType target_{TargetType::kHost};
};

}  // namespace lite
}  // namespace paddle
