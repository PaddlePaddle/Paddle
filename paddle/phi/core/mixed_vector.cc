/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/mixed_vector.h"

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <mutex>  // NOLINT
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/utils/none.h"
#include "paddle/utils/optional.h"

namespace phi {

template <typename T>
void CopyToCPUHelper(std::vector<T> *cpu_,
                     phi::Allocator::AllocationPtr *gpu_,
                     size_t *gpu_memory_size_) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  // COPY GPU Data To CPU
  auto *dev_ctx = static_cast<phi::GPUContext *>(
      phi::DeviceContextPool::Instance().Get((*gpu_)->place()));
  auto stream = dev_ctx->stream();
  void *src = (*gpu_)->ptr();
  void *dst = cpu_->data();
  memory_utils::Copy(phi::CPUPlace(),
                     dst,
                     OptionalCUDAPlace(*gpu_).get(),
                     src,
                     *gpu_memory_size_,
                     stream);
  dev_ctx->Wait();
#endif
}

template <typename T>
void CopyCPUDataToCUDAHelper(std::vector<T> *cpu_,
                             phi::Allocator::AllocationPtr *gpu_,
                             size_t *gpu_memory_size_,
                             const phi::Place &place) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void *src = cpu_->data();
  *gpu_memory_size_ = cpu_->size() * sizeof(T);  // sizeof(T)
  (*gpu_) = memory_utils::Alloc(place, *gpu_memory_size_);
  void *dst = (*gpu_)->ptr();
  auto *dev_ctx = static_cast<phi::GPUContext *>(
      phi::DeviceContextPool::Instance().Get(place));
  auto stream = dev_ctx->stream();
  memory_utils::Copy(OptionalCUDAPlace(*gpu_).get(),
                     dst,
                     phi::CPUPlace(),
                     src,
                     *gpu_memory_size_,
                     stream);
  dev_ctx->Wait();
#endif
}

#define INSTANTIATE_VECTOR_FOR_TYPE(__TYPE__)                                 \
  template <>                                                                 \
  void MixVector<__TYPE__>::VectorData::CopyToCPU() const {                   \
    CopyToCPUHelper<__TYPE__>(cpu_, &gpu_, &gpu_memory_size_);                \
  }                                                                           \
                                                                              \
  template <>                                                                 \
  void MixVector<__TYPE__>::VectorData::CopyCPUDataToCUDA(                    \
      const phi::Place &place) const {                                        \
    CopyCPUDataToCUDAHelper<__TYPE__>(cpu_, &gpu_, &gpu_memory_size_, place); \
  }

INSTANTIATE_VECTOR_FOR_TYPE(size_t)
INSTANTIATE_VECTOR_FOR_TYPE(int)
INSTANTIATE_VECTOR_FOR_TYPE(int64_t)

};  // namespace phi
