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

#include "glog/logging.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/CUDAStream.h"
#include "paddle/fluid/distributed/collective/deep_ep/include/ScalarType.h"
#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/memory/malloc.h"

namespace deep_ep::detail {

struct Tensor {
  paddle::Tensor raw_tensor_;

  explicit Tensor(const paddle::Tensor &t) : raw_tensor_(t) {}
  Tensor() : raw_tensor_() {}
  Tensor(const Tensor &) = default;
  Tensor(Tensor &&) = default;
  Tensor operator=(const Tensor &x) &noexcept {
    raw_tensor_ = x.raw_tensor_;
    return *this;
  }

  decltype(auto) raw_tensor() const { return raw_tensor_; }

  decltype(auto) dtype() const { return raw_tensor_.dtype(); }

  decltype(auto) place() const { return raw_tensor_.place(); }

  int64_t dim() const { return raw_tensor_.dims().size(); }

  bool is_contiguous() const { return true; }

  int64_t size(int64_t d) const { return raw_tensor_.dims().at(d); }

  template <typename T>
  T *data_ptr() const {
    return const_cast<T *>(raw_tensor_.data<T>());
  }

  void *data_ptr() const { return const_cast<void *>(raw_tensor_.data()); }

  template <typename T>
  T *data_ptr() {
    return raw_tensor_.data<T>();
  }

  void *data_ptr() { return raw_tensor_.data(); }

  void record_stream(const cudaStream_t &stream) const {
    paddle::memory::RecordStream(
        std::dynamic_pointer_cast<phi::DenseTensor>(raw_tensor_.impl())
            ->Holder(),
        stream);
  }

  deep_ep::detail::ScalarType scalar_type() const {
    return raw_tensor_.dtype();
  }

  int64_t element_size() const { return phi::SizeOf(raw_tensor_.dtype()); }
};

}  // namespace deep_ep::detail
