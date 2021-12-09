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

class SparseCsrTensor : public TensorBase,
                        public TypeInfoTraits<TensorBase, SparseCsrTensor> {
 public:
  SparseCsrTensor() = default;
  SparseCsrTensor(SparseCsrTensor&& other) = default;
  SparseCsrTensor(const SparseCsrTensor& other) = delete;
  SparseCsrTensor(std::unique_ptr<DenseTensor> non_zero_crows,
                  std::unique_ptr<DenseTensor> non_zero_cols,
                  std::unique_ptr<DenseTensor> non_zero_elements,
                  const DDim& dims);
  virtual ~SparseCsrTensor() = default;

  const DenseTensor& non_zero_crows() { return *non_zero_crows_; }
  const DenseTensor& non_zero_cols() { return *non_zero_cols_; }
  const DenseTensor& non_zero_elements() { return *non_zero_elements_; }

  static const char* name() { return "SparseCsrTensor"; }
  int64_t nnz() const;

  int64_t numel() const { return product(dims_); }

  const DDim& dims() const noexcept override { return dims_; }

  DataType dtype() const noexcept override {
    return non_zero_elements_->dtype();
  }

  DataLayout layout() const { return DataLayout::SPARSE_CSR; }

  const Place& place() const override { return non_zero_elements_->place(); }
  bool valid() const noexcept { return non_zero_elements_->valid(); }
  bool initialized() const override {
    return non_zero_elements_->initialized();
  }

  void SetMemberTensor(std::unique_ptr<DenseTensor> non_zero_crows,
                       std::unique_ptr<DenseTensor> non_zero_cols,
                       std::unique_ptr<DenseTensor> non_zero_elements,
                       const DDim& dims);

 private:
  std::unique_ptr<DenseTensor> non_zero_crows_;
  std::unique_ptr<DenseTensor> non_zero_cols_;
  std::unique_ptr<DenseTensor> non_zero_elements_;
  DDim dims_;
};

}  // namespace pten
