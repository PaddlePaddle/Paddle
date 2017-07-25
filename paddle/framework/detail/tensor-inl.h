/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "paddle/memory/memcpy.h"

namespace paddle {
namespace framework {

template <typename T>
inline void Tensor::check_memory_size() const {
  PADDLE_ENFORCE(holder_ != nullptr,
                 "Tenosr holds no memory. Call Tensor::mutable_data first.");
  PADDLE_ENFORCE(holder_->size() >= product(dims_) * sizeof(T) + offset_,
                 "Tensor's dims_ is out of bound. Call Tensor::mutable_data "
                 "first to re-allocate memory.");
}

template <typename T>
inline const T* Tensor::data() const {
  check_memory_size<T>();
  return reinterpret_cast<const T*>(
      reinterpret_cast<uintptr_t>(holder_->ptr()) + offset_);
}

template <typename T>
inline T* Tensor::data() {
  check_memory_size<T>();
  return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                              offset_);
}

template <typename T>
inline T* Tensor::mutable_data(DDim dims, platform::Place place) {
  static_assert(std::is_pod<T>::value, "T must be POD");
  Resize(dims);
  return mutable_data<T>(place);
}

template <typename T>
inline T* Tensor::mutable_data(platform::Place place) {
  static_assert(std::is_pod<T>::value, "T must be POD");
  PADDLE_ENFORCE(product(dims_) > 0,
                 "Tensor's numel must be larger than zero to call "
                 "Tensor::mutable_data. Call Tensor::set_dim first.");
  /* some versions of boost::variant don't have operator!= */
  size_t size = product(dims_) * sizeof(T);
  if (holder_ == nullptr || !(holder_->place() == place) ||
      holder_->size() < size + offset_) {
    if (platform::is_cpu_place(place)) {
      holder_.reset(new PlaceholderImpl<T, platform::CPUPlace>(
          boost::get<platform::CPUPlace>(place), size));
    }
#ifndef PADDLE_ONLY_CPU
    else if (platform::is_gpu_place(place)) {
      holder_.reset(new PlaceholderImpl<T, platform::GPUPlace>(
          boost::get<platform::GPUPlace>(place), size));
    }
#endif
    offset_ = 0;
  }
  return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                              offset_);
}

template <typename T>
inline void Tensor::ShareDataWith(const Tensor& src) {
  src.check_memory_size<T>();
  *this = src;
}

template <typename T>
inline void Tensor::CopyFrom(const Tensor& src,
                             const platform::CPUDeviceContext& ctx) {
  src.check_memory_size<T>();
  Resize(src.dims());

  auto src_place = src.holder_->place();
  auto src_ptr = static_cast<const void*>(src.data<T>());

  auto dst_place = ctx.GetPlace();
  auto dst_ptr = static_cast<void*>(mutable_data<T>(dst_place));

  auto size = product(src.dims_) * sizeof(T);

  if (platform::is_cpu_place(src_place)) {
    memory::Copy(boost::get<platform::CPUPlace>(dst_place), dst_ptr,
                 boost::get<platform::CPUPlace>(src_place), src_ptr, size);
  }
#ifndef PADDLE_ONLY_CPU
  else if (platform::is_gpu_place(src_place)) {
    memory::Copy(boost::get<platform::CPUPlace>(dst_place), dst_ptr,
                 boost::get<platform::GPUPlace>(src_place), src_ptr, size, 0);
  }
#endif
}

#ifndef PADDLE_ONLY_CPU
template <typename T>
inline void Tensor::CopyFrom(const Tensor& src,
                             const platform::CUDADeviceContext& ctx) {
  src.check_memory_size<T>();
  Resize(src.dims());

  auto src_place = src.holder_->place();
  auto src_ptr = static_cast<const void*>(src.data<T>());

  auto dst_place = ctx.GetPlace();
  auto dst_ptr = static_cast<void*>(mutable_data<T>(dst_place));

  auto size = product(src.dims_) * sizeof(T);

  if (platform::is_cpu_place(src_place)) {
    memory::Copy(boost::get<platform::GPUPlace>(dst_place), dst_ptr,
                 boost::get<platform::CPUPlace>(src_place), src_ptr, size,
                 ctx.stream());
  } else if (platform::is_gpu_place(src_place)) {
    memory::Copy(boost::get<platform::GPUPlace>(dst_place), dst_ptr,
                 boost::get<platform::GPUPlace>(src_place), src_ptr, size,
                 ctx.stream());
  }
}
#endif

template <typename T>
inline Tensor Tensor::Slice(const int& begin_idx, const int& end_idx) const {
  check_memory_size<T>();
  PADDLE_ENFORCE(begin_idx >= 0, "Slice begin index is less than zero.");
  PADDLE_ENFORCE(end_idx <= dims_[0], "Slice end index is out of bound.");
  PADDLE_ENFORCE(begin_idx < end_idx,
                 "Begin index must be less than end index.");
  PADDLE_ENFORCE(dims_[0] != 1, "Can not slice a tensor with dims_[0] = 1.");
  int base = product(dims_) / dims_[0];
  Tensor dst;
  dst.holder_ = holder_;
  DDim dst_dims = dims_;
  dst_dims[0] = end_idx - begin_idx;
  dst.Resize(dst_dims);
  dst.offset_ = offset_ + begin_idx * base * sizeof(T);
  return dst;
}

inline void Tensor::Resize(const DDim& dims) { dims_ = dims; }

inline const DDim& Tensor::dims() const { return dims_; }

}  // namespace framework
}  // namespace paddle
