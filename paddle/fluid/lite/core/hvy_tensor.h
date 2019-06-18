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

/*
 * This file defines the heavy tensor (alias of the LoDTensor in the server
 * framework). We derive it from the TensorLite interface, so the lite framework
 * can share much code between the server side and mobile side.
 */

#pragma once
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/lite/core/target_wrapper.h"
#include "paddle/fluid/lite/core/tensor.h"

namespace paddle {
namespace lite {

class DDimHvy : public DDimBase<DDimHvy> {
 public:
  DDimHvy() = default;
  DDimHvy(const std::vector<value_type>& x) : DDimBase<DDimHvy>() {  // NOLINT
    ConstructFrom(x);
  }
  explicit DDimHvy(const framework::DDim& x) : data_(x) {}

  void ConstructFrom(const std::vector<value_type>& xs) {
    data_ = framework::DDim(xs.data(), xs.size());
  }

  value_type operator[](int offset) const { return data_[offset]; }
  value_type& operator[](int offset) { return data_[offset]; }

  std::vector<int64_t> Vectorize() const { return framework::vectorize(data_); }

  const framework::DDim& data() const { return data_; }

  size_t size() const { return data_.size(); }
  bool empty() const { return data_.size() == 0; }

  bool operator==(const DDimHvy& other) {
    if (data_.size() != other.data_.size()) return false;
    for (int i = 0; i < data_.size(); i++) {
      if (data_[i] != other.data_[i]) return false;
    }
    return true;
  }

 private:
  framework::DDim data_;
};

class TensorHvy : public TensorBase<TensorHvy> {
 public:
  using DDimT = DDimHvy;
  using LoDT = framework::LoD;

  template <typename DType, typename DimT, TargetType Target>
  void Assign(DType* data, const DimT& dim) {
    Resize(dim);
    auto* dst = mutable_data<DType>(Target);
    CopySync<Target>(dst, data, dim.production() * sizeof(DType),
                     IoDirection::HtoD);
  }

  TargetType target() const {
    if (platform::is_gpu_place(data_.place())) {
      return TARGET(kCUDA);
    } else if (platform::is_cpu_place(data_.place())) {
      return TARGET(kX86);
    }
    LOG(FATAL) << "Unknown place";
    return TARGET(kUnk);
  }

  template <typename T>
  T* mutable_data() {
    memory_size_ = framework::product(data_.dims()) * sizeof(T);
    return data_.mutable_data<T>(data_.dims(), platform::CPUPlace());
  }
  template <typename T>
  T* mutable_data(TargetType target) {
    if (target == TARGET(kCUDA)) {
      return data_.mutable_data<T>(data_.dims(), platform::CUDAPlace());
    }
    return data_.mutable_data<T>(data_.dims(), platform::CPUPlace());
  }

  template <typename T>
  const T* data() const {
    return data_.data<T>();
  }

  const void* raw_data() const { return data_.raw_data(); }

  void Resize(const DDimHvy& dims) {
    data_.Resize(framework::make_ddim(dims.Vectorize()));
  }

  void Resize(const std::vector<int64_t>& x) { Resize(DDimHvy(x)); }

  void ShareDataWith(const TensorHvy& other) {
    data_.ShareDataWith(other.data_);
  }
  void ShareDataWith(const framework::Tensor& other) {
    data_.ShareDataWith(other);
  }
  void CopyDataFrom(const TensorHvy& other) {
    data_.mutable_data(other.data_.place(), other.data_.type());
    TensorCopySync(other.data_, data_.place(), &data_);
  }

  DDimT dims() const { return DDimT(framework::vectorize(data_.dims())); }

  const framework::LoD& lod() const { return data_.lod(); }
  framework::LoD* mutable_lod() { return data_.mutable_lod(); }

  const framework::LoDTensor& raw_tensor() const { return data_; }
  framework::LoDTensor& raw_tensor() { return data_; }

  size_t memory_size() const { return memory_size_; }

 private:
  framework::LoDTensor data_;
  size_t memory_size_{};
};

}  // namespace lite
}  // namespace paddle
