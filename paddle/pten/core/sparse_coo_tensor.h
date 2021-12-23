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

#include "paddle/fluid/framework/rw_lock.h"
#include "paddle/pten/core/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/tensor_base.h"
#include "paddle/pten/core/tensor_meta.h"

namespace pten {

class CompatibleDenseTensorUtils;
using RWLock = paddle::framework::RWLock;

class SparseCooTensor : public TensorBase,
                        public TypeInfoTraits<TensorBase, SparseCooTensor> {
 public:
  SparseCooTensor() = default;
  SparseCooTensor(const std::shared_ptr<Allocator>& a,
                  const DenseTensorMeta& dense_meta);
  SparseCooTensor(const DenseTensor& non_zero_indices,
                  const DenseTensor& non_zero_elements,
                  const DDim& dims);
  SparseCooTensor(DenseTensor&& non_zero_indices,
                  DenseTensor&& non_zero_elements,
                  const DDim& dims);
  /// \brief SparseCsrTensor shallow copy constructor.
  SparseCooTensor(const SparseCooTensor& other);

  virtual ~SparseCooTensor() = default;

  const DenseTensor& non_zero_indices() const { return non_zero_indices_; }
  const DenseTensor& non_zero_elements() const { return non_zero_elements_; }

  int64_t sparse_dim() { return sparse_dim_; }
  int64_t dense_dim() { return dense_dim_; }
  bool coalesced() { return coalesced_; }

  static const char* name() { return "SparseCooTensor"; }

  int64_t nnz() const;

  int64_t numel() const { return product(dims_); }

  const DDim& dims() const noexcept override { return dims_; }

  DataType dtype() const noexcept override {
    return non_zero_elements_.dtype();
  }

  DataLayout layout() const { return DataLayout::SPARSE_COO; }

  const Place& place() const override { return non_zero_elements_.place(); }
  bool valid() const noexcept { return non_zero_elements_.valid(); }
  bool initialized() const override { return non_zero_elements_.initialized(); }

  void SetMember(const DenseTensor& non_zero_indices,
                 const DenseTensor& non_zero_elements,
                 const DDim& dims);

  void Resize(const DDim& dense_dim,
              const int64_t sparse_dim,
              const int64_t non_zero_num);

  void Resize(const std::shared_ptr<Allocator>& a,
              const DenseTensorMeta& meta,
              const int64_t non_zero_num);

  int64_t* mutable_non_zero_indices() {
    return non_zero_indices_.mutable_data<int64_t>();
  }
  template <typename T>
  T* mutable_non_zero_elements() {
    return non_zero_elements_.mutable_data<T>();
  }

  int64_t Index(const std::vector<int64_t>& indices) const;
  int64_t Index(int64_t indices) const;

  int64_t AutoGrownIndex(const std::vector<int64_t>& indices,
                         bool auto_grown,
                         bool is_test = false) const;
  int64_t AutoGrownIndex(int64_t indices,
                         bool auto_grown,
                         bool is_test = false) const;

  void SyncIndex();

 private:
  int64_t sparse_dim_ = 0;
  int64_t dense_dim_ = 0;
  DenseTensor non_zero_indices_;
  DenseTensor non_zero_elements_;
  bool coalesced_ = false;
  DDim dims_;

  std::map<int64_t, int64_t> indices_to_index_;
  int64_t height_;  // height indicates the underline tensor's height
  std::unique_ptr<RWLock> rwlock_{nullptr};
};

}  // namespace pten
