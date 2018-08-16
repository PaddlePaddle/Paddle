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

#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace framework {
extern size_t SizeOfType(std::type_index type);
void Tensor::check_memory_size() const {
  PADDLE_ENFORCE_NOT_NULL(
      holder_, "Tensor holds no memory. Call Tensor::mutable_data first.");
  PADDLE_ENFORCE_LE(
      numel() * SizeOfType(type()), memory_size(),
      "Tensor's dims_ is out of bound. Call Tensor::mutable_data "
      "first to re-allocate memory.\n"
      "or maybe the required data-type mismatches the data already stored.");
}

size_t Tensor::memory_size() const {
  return holder_ == nullptr ? 0UL : holder_->size() - offset_;
}

void* Tensor::mutable_data(platform::Place place, std::type_index type) {
  if (holder_ != nullptr) {
    holder_->set_type(type);
  }
  PADDLE_ENFORCE_GE(numel(), 0,
                    "When calling this method, the Tensor's numel must be "
                    "equal or larger than zero. "
                    "Please check Tensor::Resize has been called first.");
  int64_t size = numel() * SizeOfType(type);
  /* some versions of boost::variant don't have operator!= */
  if (holder_ == nullptr || !(holder_->place() == place) ||
      holder_->size() < size + offset_) {
    if (platform::is_cpu_place(place)) {
      holder_.reset(new PlaceholderImpl<platform::CPUPlace>(
          boost::get<platform::CPUPlace>(place), size, type));
    } else if (platform::is_gpu_place(place) ||
               platform::is_cuda_pinned_place(place)) {
#ifndef PADDLE_WITH_CUDA
      PADDLE_THROW(
          "CUDAPlace or CUDAPinnedPlace is not supported in CPU-only mode.");
    }
#else
      if (platform::is_gpu_place(place)) {
        holder_.reset(new PlaceholderImpl<platform::CUDAPlace>(
            boost::get<platform::CUDAPlace>(place), size, type));
      } else if (platform::is_cuda_pinned_place(place)) {
        holder_.reset(new PlaceholderImpl<platform::CUDAPinnedPlace>(
            boost::get<platform::CUDAPinnedPlace>(place), size, type));
      }
    }
#endif
    offset_ = 0;
  }
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                                 offset_);
}

void* Tensor::mutable_data(platform::Place place) {
  PADDLE_ENFORCE(this->holder_ != nullptr,
                 "Cannot invoke mutable data if current hold nothing.");
  return mutable_data(place, holder_->type());
}

Tensor& Tensor::ShareDataWith(const Tensor& src) {
  src.check_memory_size();
  *this = src;
  return *this;
}

Tensor Tensor::Slice(int begin_idx, int end_idx) const {
  check_memory_size();
  PADDLE_ENFORCE_GE(begin_idx, 0,
                    "The start row index must be greater than 0.");
  PADDLE_ENFORCE_LE(end_idx, dims_[0], "The end row index is out of bound.");
  PADDLE_ENFORCE_LT(
      begin_idx, end_idx,
      "The start row index must be lesser than the end row index.");

  if (dims_[0] == 1) {
    return *this;
  } else {
    size_t base = numel() / dims_[0];
    Tensor dst;
    dst.holder_ = holder_;
    dst.set_layout(layout_);
    DDim dst_dims = dims_;
    dst_dims[0] = end_idx - begin_idx;
    dst.Resize(dst_dims);
    dst.offset_ = offset_ + begin_idx * base * SizeOfType(type());
    return dst;
  }
}

Tensor& Tensor::Resize(const DDim& dims) {
  dims_ = dims;
  return *this;
}

const DDim& Tensor::dims() const { return dims_; }

int64_t Tensor::numel() const { return product(dims_); }

}  // namespace framework
}  // namespace paddle
