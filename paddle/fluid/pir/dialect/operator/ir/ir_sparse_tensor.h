// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_meta.h"

namespace paddle {
namespace dialect {

class IrSparseCooTensor
    : public phi::TensorBase,
      public phi::TypeInfoTraits<phi::TensorBase, IrSparseCooTensor> {
 public:
  IrSparseCooTensor() = default;

  IrSparseCooTensor(phi::DataType dtype,
                    const common::DDim& dims,
                    common::DDim non_zero_dims,
                    common::DataLayout layout,
                    bool coalesced = false);
  IrSparseCooTensor(IrSparseCooTensor&& other) = default;

  IrSparseCooTensor(const IrSparseCooTensor& other) = default;

  IrSparseCooTensor& operator=(const IrSparseCooTensor& other) = default;

  IrSparseCooTensor& operator=(IrSparseCooTensor&& other) noexcept;

  virtual ~IrSparseCooTensor() = default;

 public:
  static const char* name() { return "IrSparseCooTensor"; }

  int64_t numel() const override;

  const common::DDim& dims() const noexcept override { return dims_; }

  void SetDims(const common::DDim& dims) { dims_ = dims; }

  const common::DDim& non_zero_dims() const noexcept { return non_zero_dims_; }

  void SetNonZeroDims(const common::DDim& non_zero_dims) {
    non_zero_dims_ = non_zero_dims;
  }

  const phi::Place& place() const override;

  phi::DataType dtype() const noexcept override { return dtype_; }

  void SetDtype(phi::DataType dtype) { dtype_ = dtype; }

  common::DataLayout layout() const noexcept override { return layout_; }

  void SetLayout(common::DataLayout layout) { layout_ = layout; }

  bool coalesced() const { return coalesced_; }

  void SetCoalesced(bool coalesced) { coalesced_ = coalesced; }

  bool valid() const noexcept override { return true; }

  bool has_allocation() const override {
    PADDLE_THROW(::common::errors::Unavailable(
        "`has_allocation` is only available at runtime"));
  }

  bool initialized() const override { return true; }

  void* AllocateFrom(phi::Allocator* allocator,
                     phi::DataType dtype,
                     size_t requested_size = 0,
                     bool fake_alloc = false) override;

 private:
  common::DDim dims_;
  common::DDim non_zero_dims_;
  phi::DataType dtype_{phi::DataType::FLOAT32};
  common::DataLayout layout_{common::DataLayout::ANY};
  bool coalesced_ = false;
};

class IrSparseCsrTensor
    : public phi::TensorBase,
      public phi::TypeInfoTraits<phi::TensorBase, IrSparseCsrTensor> {
 public:
  IrSparseCsrTensor() = default;

  IrSparseCsrTensor(phi::DataType dtype,
                    const common::DDim& dims,
                    common::DataLayout layout,
                    pir::DenseTensorType non_zero_crows,
                    pir::DenseTensorType non_zero_cols,
                    pir::DenseTensorType non_zero_elements);
  IrSparseCsrTensor(IrSparseCsrTensor&& other) = default;

  IrSparseCsrTensor(const IrSparseCsrTensor& other) = default;

  IrSparseCsrTensor& operator=(const IrSparseCsrTensor& other) = default;

  IrSparseCsrTensor& operator=(IrSparseCsrTensor&& other) noexcept;

  virtual ~IrSparseCsrTensor() = default;

 public:
  static const char* name() { return "IrSparseCsrTensor"; }

  int64_t numel() const override;

  const common::DDim& dims() const noexcept override { return dims_; }

  void SetDims(const common::DDim& dims) { dims_ = dims; }

  const phi::Place& place() const override;

  phi::DataType dtype() const noexcept override { return dtype_; }

  void SetDtype(phi::DataType dtype) { dtype_ = dtype; }

  common::DataLayout layout() const noexcept override { return layout_; }

  void SetLayout(common::DataLayout layout) { layout_ = layout; }

  pir::DenseTensorType non_zero_crows() const { return non_zero_crows_; }

  void SetNonZeroCrows(pir::DenseTensorType non_zero_crows) {
    non_zero_crows_ = non_zero_crows;
  }

  pir::DenseTensorType non_zero_cols() const { return non_zero_cols_; }

  void SetNonZeroCols(pir::DenseTensorType non_zero_cols) {
    non_zero_cols_ = non_zero_cols;
  }

  pir::DenseTensorType non_zero_elements() const { return non_zero_elements_; }

  void SetNonZeroElements(pir::DenseTensorType non_zero_elements) {
    non_zero_elements_ = non_zero_elements;
  }

  bool valid() const noexcept override { return true; }

  bool has_allocation() const override { return true; }

  bool initialized() const override { return true; }

  void* AllocateFrom(phi::Allocator* allocator,
                     phi::DataType dtype,
                     size_t requested_size = 0,
                     bool fake_alloc = false) override;

 private:
  common::DDim dims_;
  phi::DataType dtype_{phi::DataType::FLOAT32};
  common::DataLayout layout_{common::DataLayout::ANY};
  pir::DenseTensorType non_zero_crows_;
  pir::DenseTensorType non_zero_cols_;
  pir::DenseTensorType non_zero_elements_;
};

inline SparseCooTensorType CvtToSparseCooTensorType(
    const IrSparseCooTensor& ir_tensor) {
  pir::IrContext* ctx = pir::IrContext::Instance();
  pir::Type fp32_dtype = pir::Float32Type::get(ctx);
  phi::DataLayout data_layout = phi::DataLayout::UNDEFINED;
  phi::LegacyLoD lod = {};
  phi::DDim dims = {};
  size_t offset = 0;
  pir::DenseTensorType non_zero_indices = pir::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);

  pir::DenseTensorType non_zero_elements = pir::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);

  return SparseCooTensorType::get(pir::IrContext::Instance(),
                                  TransToIrDataType(ir_tensor.dtype()),
                                  ir_tensor.dims(),
                                  ir_tensor.non_zero_dims(),
                                  ir_tensor.layout(),
                                  non_zero_indices,
                                  non_zero_elements,
                                  ir_tensor.coalesced());
}

inline SparseCsrTensorType CvtToSparseCsrTensorType(
    const IrSparseCsrTensor& ir_tensor) {
  return SparseCsrTensorType::get(pir::IrContext::Instance(),
                                  TransToIrDataType(ir_tensor.dtype()),
                                  ir_tensor.dims(),
                                  ir_tensor.layout(),
                                  ir_tensor.non_zero_crows(),
                                  ir_tensor.non_zero_cols(),
                                  ir_tensor.non_zero_elements());
}
}  // namespace dialect
}  // namespace paddle
