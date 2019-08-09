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
#include <algorithm>
#include <functional>  // for multiplies
#include <memory>
#include <numeric>
#include <vector>

#include "paddle/fluid/lite/core/memory.h"
#include "paddle/fluid/lite/core/target_wrapper.h"
#include "paddle/fluid/lite/core/tensor.h"

namespace paddle {
namespace lite {

class DDimLite : public DDimBase<DDimLite> {
 public:
  DDimLite() = default;

  explicit DDimLite(const std::vector<value_type> &x) : DDimBase<DDimLite>() {
    ConstructFrom(x);
  }

  void ConstructFrom(const std::vector<value_type> &x) { data_ = x; }

  value_type operator[](int offset) const { return data_[offset]; }
  value_type &operator[](int offset) { return data_[offset]; }
  std::vector<int64_t> Vectorize() const { return data_; }

  size_t size() const { return data_.size(); }
  bool empty() const { return data_.empty(); }
  value_type product() const {
    return std::accumulate(std::begin(data_), std::end(data_), 1,
                           std::multiplies<value_type>());
  }
  const std::vector<value_type> &data() const { return data_; }
  value_type count(int start, int end) const {
    if (start < 0) {
      start = 0;
    }
    if (end > size()) {
      end = size();
    }
    if (end < start) {
      end = start;
    }
    value_type sum = 1;
    for (auto i = start; i < end; ++i) {
      sum *= data_[i];
    }
    return sum;
  }

 private:
  std::vector<value_type> data_;
};

using LoD = std::vector<std::vector<size_t>>;

// A light-weight tensor implementation.
class TensorLite : public TensorBase<TensorLite> {
 public:
  using DDimT = DDimLite;

  TensorLite() : buffer_(std::make_shared<Buffer>()) {}

  template <typename DType, typename DimT, TargetType Target>
  void Assign(DType *data, const DimT &dim) {
    Resize(dim);
    auto *dst = mutable_data<DType>(Target);
    CopySync<Target>(dst, data, dim.product() * sizeof(DType),
                     IoDirection::HtoD);
  }

  template <typename T>
  const T *data() const {
    return static_cast<const T *>(buffer_->data());
  }

  void Resize(const DDimLite &ddim) { dims_ = ddim; }
  void Resize(const std::vector<int64_t> &x) { dims_ = DDimLite(x); }

  const DDimLite &dims() const { return dims_; }
  int64_t numel() const { return dims_.product(); }

  const LoD &lod() const { return lod_; }
  LoD *mutable_lod() { return &lod_; }

  template <typename T>
  T *mutable_data();
  template <typename T>
  T *mutable_data(TargetType target);
  void *mutable_data(size_t memory_size);
  void *mutable_data(TargetType target, size_t memory_size);

  const void *raw_data() const { return buffer_->data(); }

  size_t memory_size() const { return memory_size_; }

  bool IsInitialized() const { return buffer_->data(); }

  // Other share data to this.
  void ShareDataWith(const TensorLite &other);

  void CopyDataFrom(const TensorLite &other);

  TargetType target() const { return target_; }

 private:
  TargetType target_{TargetType::kHost};
  DDimLite dims_;
  std::shared_ptr<Buffer> buffer_;
  LoD lod_;
  size_t memory_size_{};
};

template <typename T>
T *TensorLite::mutable_data() {
  memory_size_ = dims_.production() * sizeof(T);
  buffer_->ResetLazy(target_, memory_size_);
  return static_cast<T *>(buffer_->data());
}

template <typename T>
T *TensorLite::mutable_data(TargetType target) {
  target_ = target;
  memory_size_ = dims_.production() * sizeof(T);
  buffer_->ResetLazy(target, memory_size());
  return static_cast<T *>(buffer_->data());
}

}  // namespace lite
}  // namespace paddle
