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

#include "paddle/fluid/lite/core/memory.h"

namespace paddle {
namespace lite {

void* TargetMalloc(TargetType target, size_t size) {
  void* data{nullptr};
  switch (target) {
    case TargetType::kHost:
    case TargetType::kX86:
    case TargetType::kARM:
      data = TargetWrapper<TARGET(kHost)>::Malloc(size);
      break;
#ifdef LITE_WITH_CUDA
    case TargetType::kCUDA:
      data =
          TargetWrapper<TARGET(kCUDA), cudaStream_t, cudaEvent_t>::Malloc(size);
      break;
#endif  // LITE_WITH_CUDA
    default:
      LOG(FATAL) << "Unknown supported target " << TargetToStr(target);
  }
  return data;
}

void TargetFree(TargetType target, void* data) {
  switch (target) {
    case TargetType::kHost:
    case TargetType::kX86:
    case TargetType::kARM:
      TargetWrapper<TARGET(kHost)>::Free(data);
      break;

#ifdef LITE_WITH_CUDA
    case TargetType::kCUDA:
      TargetWrapper<TARGET(kCUDA)>::Free(data);
      break;
#endif  // LITE_WITH_CUDA
    default:
      LOG(FATAL) << "Unknown type";
  }
}

void TargetCopy(TargetType target, void* dst, const void* src, size_t size) {
  switch (target) {
    case TargetType::kHost:
    case TargetType::kX86:
    case TargetType::kARM:
      TargetWrapper<TARGET(kHost)>::MemcpySync(dst, src, size,
                                               IoDirection::DtoD);
      break;

#ifdef LITE_WITH_CUDA
    case TargetType::kCUDA:
      TargetWrapper<TARGET(kCUDA)>::MemcpySync(dst, src, size,
                                               IoDirection::DtoD);
      break;
#endif
    default:
      LOG(FATAL) << "unsupported type";
  }
}

}  // namespace lite
}  // namespace paddle
