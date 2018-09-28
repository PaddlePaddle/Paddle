// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/memory_holder.h"

namespace paddle {
namespace framework {
namespace internal {

void MemoryHolder::Clear() {
  holder_.reset();
  offset_ = 0;
}

const void *MemoryHolder::Get() const {
  size_t size = GetMemorySize();
  PADDLE_ENFORCE(holder_ == nullptr || size + offset_ <= holder_->size_);
  return size > 0 ? reinterpret_cast<void *>(
                        reinterpret_cast<uintptr_t>(holder_->ptr_) + offset_)
                  : nullptr;
}

void *MemoryHolder::GetMutable(const platform::Place &place, CopyType copy) {
  size_t size = GetMemorySize();

  if (UNLIKELY(size == 0)) {
    holder_.reset();
    offset_ = 0;
    return nullptr;
  }

  if (holder_ == nullptr) {
    holder_.reset(new InternalHolder(size, place));
    offset_ = 0;
    return holder_->ptr_;
  }

  bool has_enough_space = (size + offset_ <= holder_->size_);

  if (holder_->place_ == place && has_enough_space &&
      holder_.use_count() == 1) {
    return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(holder_->ptr_) +
                                    offset_);
  } else {
    auto new_holder = new InternalHolder(size, place);

    if (copy != kNone) {
      void *dst = new_holder->ptr_;
      const void *src = reinterpret_cast<void *>(
          reinterpret_cast<uintptr_t>(holder_->ptr_) + offset_);
      // TODO(zjl): do we really need to copy if has_enough_space is false?
      size_t copy_num = has_enough_space
                            ? size
                            : static_cast<size_t>(holder_->size_ - offset_);

// TODO(zjl): The following code is rather ugly
// copy from src (holder_->place_) to dst (place)

#define PADDLE_MEMCPY_BUF(dst_place, src_place)           \
  memory::Copy<platform::dst_place, platform::src_place>( \
      boost::get<platform::dst_place>(place), dst,        \
      boost::get<platform::src_place>(holder_->place_), src, copy_num)

#ifdef PADDLE_WITH_CUDA
#define PADDLE_MEMCPY_BUF_WITH_STREAM(dst_place, src_place, stream) \
  memory::Copy<platform::dst_place, platform::src_place>(           \
      boost::get<platform::dst_place>(place), dst,                  \
      boost::get<platform::src_place>(holder_->place_), src, copy_num, stream)
#endif

#ifdef PADDLE_WITH_CUDA
      if (platform::is_gpu_place(place)) {
        auto stream =
            (copy == kSync
                 ? nullptr
                 : static_cast<platform::CUDADeviceContext *>(
                       platform::DeviceContextPool::Instance().Get(place))
                       ->stream());
        if (platform::is_gpu_place(holder_->place_)) {
          if (holder_->place_ == place) {
            PADDLE_MEMCPY_BUF_WITH_STREAM(CUDAPlace, CUDAPlace, stream);
          } else {
            platform::DeviceContextPool::Instance()
                .Get(holder_->place_)
                ->Wait();
            PADDLE_MEMCPY_BUF_WITH_STREAM(CUDAPlace, CUDAPlace, stream);
          }
        } else if (platform::is_cpu_place(holder_->place_)) {
          PADDLE_MEMCPY_BUF_WITH_STREAM(CUDAPlace, CPUPlace, stream);
        } else {
          PADDLE_MEMCPY_BUF_WITH_STREAM(CUDAPlace, CUDAPinnedPlace, stream);
        }
      } else if (platform::is_gpu_place(holder_->place_)) {
        auto stream =
            (copy == kSync ? nullptr
                           : static_cast<platform::CUDADeviceContext *>(
                                 platform::DeviceContextPool::Instance().Get(
                                     holder_->place_))
                                 ->stream());
        if (platform::is_cpu_place(place)) {
          PADDLE_MEMCPY_BUF_WITH_STREAM(CPUPlace, CUDAPlace, stream);
        } else {
          PADDLE_MEMCPY_BUF_WITH_STREAM(CUDAPinnedPlace, CUDAPlace, stream);
        }
      } else {
#else
      if (platform::is_cpu_place(place)) {
        if (platform::is_cpu_place(holder_->place_)) {
          PADDLE_MEMCPY_BUF(CPUPlace, CPUPlace);
        } else {
          PADDLE_MEMCPY_BUF(CPUPlace, CUDAPinnedPlace);
        }
      } else {
        if (platform::is_cpu_place(holder_->place_)) {
          PADDLE_MEMCPY_BUF(CUDAPinnedPlace, CPUPlace);
        } else {
          PADDLE_MEMCPY_BUF(CUDAPinnedPlace, CUDAPinnedPlace);
        }
      }
#endif
#ifdef PADDLE_WITH_CUDA
      }
#endif
    }

    holder_.reset(new_holder);
    offset_ = 0;
    return holder_->ptr_;
  }
}

#undef PADDLE_MEMCPY_BUF

#ifdef PADDLE_WITH_CUDA
#undef PADDLE_MEMCPY_BUF_WITH_STREAM
#endif

}  // namespace internal
}  // namespace framework
}  // namespace paddle
