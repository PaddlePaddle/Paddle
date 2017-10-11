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
#include "paddle/framework/tensor.h"
#ifdef PADDLE_WITH_CUDA
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#endif

namespace paddle {
namespace framework {

#ifndef PADDLE_WITH_CUDA
template <typename T>
using Vector = std::vector<T>;
#else
template <typename T>
using Vector = thrust::host_vector<
    T, thrust::system::cuda::experimental::pinned_allocator<T>>;
#endif

class SelectedRows {
 public:
  SelectedRows(const Vector<int64_t>& rows, const int64_t& height)
      : rows_(rows), height_(height) {}

  void set_value(Tensor* value) { value_ = value; }

  platform::Place place() const { return value_->place(); }

  const Tensor& value() const { return *value_; }

  int64_t height() const { return height_; }

  const Vector<int64_t>& rows() const { return rows_; }

 private:
  Vector<int64_t> rows_;
  Tensor* value_;  // not owned
  int64_t height_;
};

template <typename T>
void SelectedRowsToTensor(const SelectedRows& input,
                          const platform::Place& dst_place,
                          platform::DeviceContext& ctx, Tensor* output);

}  // namespace framework
}  // namespace paddle
