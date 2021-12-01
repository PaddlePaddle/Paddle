/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/pten/core/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/tensor_base.h"
#include "paddle/pten/core/tensor_meta.h"

namespace pten {

class CompatibleDenseTensorUtils;

class SparseCooTensor : public TensorBase,
                        public TypeInfoTraits<TensorBase, SparseCooTensor> {
 public:
  SparseCooTensor() = default;
  SparseCooTensor(const std::shared_ptr<Allocator>& a,
                  const DenseTensorMeta& dense_meta);
  SparseCooTensor(std::unique_ptr<DenseTensor> indices,
                  std::unique_ptr<DenseTensor> values,
                  const DDim& dims);
  virtual ~SparseCooTensor() = default;

  const DenseTensor& indices() { return *indices_; }
  const DenseTensor& values() { return *values_; }

  int64_t* mutable_indices() { return indices_->mutable_data<int64_t>(); }

  template <typename T>
  T* mutable_values() {
    return values_->mutable_data<T>();
  }

  int64_t sparse_dim() { return sparse_dim_; }
  int64_t dense_dim() { return dense_dim_; }
  bool coalesced() { return coalesced_; }

  static const char* name() { return "SparseCooTensor"; }

  int64_t numel() const { return values_->numel(); }

  const DDim& dims() const noexcept override { return dims_; }

  DataType dtype() const noexcept override { return values_->dtype(); }

  DataLayout layout() const { return DataLayout::SPARSE_COO; }

  const Place& place() const override { return values_->place(); }
  bool valid() const noexcept { return values_->valid(); }
  bool initialized() const override { return values_->initialized(); }

  void set_indices_and_values_unsafe(std::unique_ptr<DenseTensor> indices,
                                     std::unique_ptr<DenseTensor> values,
                                     const DDim& dims);

 private:
  int64_t sparse_dim_ = 0;
  int64_t dense_dim_ = 0;
  std::unique_ptr<DenseTensor> indices_;
  std::unique_ptr<DenseTensor> values_;
  bool coalesced_ = false;
  DDim dims_;
};

}  // namespace pten
