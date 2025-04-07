// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pir/dialect/kernel/ir/type_storage.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/include/core/type.h"

namespace paddle {
namespace dialect {

class AllocatedDenseTensorType
    : public pir::Type::TypeBase<AllocatedDenseTensorType,
                                 pir::Type,
                                 AllocatedDenseTensorTypeStorage,
                                 pir::WrapTypeInterface> {
 public:
  using Base::Base;

  static AllocatedDenseTensorType get(pir::IrContext *ctx,
                                      const phi::Place &place,
                                      dialect::DenseTensorType type) {
    return pir::TypeManager::template get<AllocatedDenseTensorType>(
        ctx, place, type);
  }

  static AllocatedDenseTensorType get(pir::IrContext *ctx,
                                      const phi::Place &place,
                                      const pir::Type &dtype,
                                      const phi::DDim &dims,
                                      const phi::DataLayout &layout,
                                      const phi::LegacyLoD &lod,
                                      size_t offset) {
    dialect::DenseTensorType dense_tensor_type =
        dialect::DenseTensorType::get(ctx, dtype, dims, layout, lod, offset);

    return pir::TypeManager::template get<AllocatedDenseTensorType>(
        ctx, place, dense_tensor_type);
  }

  pir::Type prim_type();

  const phi::Place &place() const;

  pir::Type dtype() const;

  const phi::DDim &dims() const;

  phi::DataLayout data_layout() const;

  const phi::LegacyLoD &lod() const;

  size_t offset() const;
};

class AllocatedSelectedRowsType
    : public pir::Type::TypeBase<AllocatedSelectedRowsType,
                                 pir::Type,
                                 AllocatedSelectedRowsTypeStorage,
                                 pir::WrapTypeInterface> {
 public:
  using Base::Base;

  static AllocatedSelectedRowsType get(pir::IrContext *ctx,
                                       const phi::Place &place,
                                       dialect::SelectedRowsType type) {
    return pir::TypeManager::template get<AllocatedSelectedRowsType>(
        ctx, place, type);
  }

  static AllocatedSelectedRowsType get(pir::IrContext *ctx,
                                       const phi::Place &place,
                                       const pir::Type &dtype,
                                       const phi::DDim &dims,
                                       const phi::DataLayout &layout,
                                       const phi::LegacyLoD &lod,
                                       size_t offset) {
    dialect::SelectedRowsType type =
        dialect::SelectedRowsType::get(ctx, dtype, dims, layout, lod, offset);

    return pir::TypeManager::template get<AllocatedSelectedRowsType>(
        ctx, place, type);
  }

  pir::Type prim_type();

  const phi::Place &place() const;

  pir::Type dtype() const;

  const phi::DDim &dims() const;

  phi::DataLayout data_layout() const;

  const phi::LegacyLoD &lod() const;

  size_t offset() const;
};

class AllocatedSparseCooTensorType
    : public pir::Type::TypeBase<AllocatedSparseCooTensorType,
                                 pir::Type,
                                 AllocatedSparseCooTensorTypeStorage,
                                 pir::WrapTypeInterface> {
 public:
  using Base::Base;

  static AllocatedSparseCooTensorType get(pir::IrContext *ctx,
                                          const phi::Place &place,
                                          dialect::SparseCooTensorType type) {
    return pir::TypeManager::template get<AllocatedSparseCooTensorType>(
        ctx, place, type);
  }

  static AllocatedSparseCooTensorType get(
      pir::IrContext *ctx,
      const phi::Place &place,
      const pir::Type &dtype,
      const phi::DDim &dims,
      const phi::DDim &non_zero_dims,
      const phi::DataLayout &layout,
      pir::DenseTensorType non_zero_indices,
      pir::DenseTensorType non_zero_elements,
      bool coalesced) {
    dialect::SparseCooTensorType type =
        dialect::SparseCooTensorType::get(ctx,
                                          dtype,
                                          dims,
                                          non_zero_dims,
                                          layout,
                                          non_zero_indices,
                                          non_zero_elements,
                                          coalesced);

    return pir::TypeManager::template get<AllocatedSparseCooTensorType>(
        ctx, place, type);
  }

  pir::Type prim_type();

  const phi::Place &place() const;

  const pir::Type dtype() const;

  const phi::DDim &dims() const;

  const phi::DDim &non_zero_dims() const;

  phi::DataLayout data_layout() const;

  pir::DenseTensorType non_zero_indices() const;

  pir::DenseTensorType non_zero_elements() const;

  bool coalesced() const;
};

class AllocatedSparseCsrTensorType
    : public pir::Type::TypeBase<AllocatedSparseCsrTensorType,
                                 pir::Type,
                                 AllocatedSparseCsrTensorTypeStorage,
                                 pir::WrapTypeInterface> {
 public:
  using Base::Base;

  static AllocatedSparseCsrTensorType get(pir::IrContext *ctx,
                                          const phi::Place &place,
                                          dialect::SparseCsrTensorType type) {
    return pir::TypeManager::template get<AllocatedSparseCsrTensorType>(
        ctx, place, type);
  }

  static AllocatedSparseCsrTensorType get(
      pir::IrContext *ctx,
      const phi::Place &place,
      const pir::Type &dtype,
      const phi::DDim &dims,
      const phi::DataLayout &layout,
      pir::DenseTensorType non_zero_crows,
      pir::DenseTensorType non_zero_cols,
      pir::DenseTensorType non_zero_elements) {
    dialect::SparseCsrTensorType type =
        dialect::SparseCsrTensorType::get(ctx,
                                          dtype,
                                          dims,
                                          layout,
                                          non_zero_crows,
                                          non_zero_cols,
                                          non_zero_elements);

    return pir::TypeManager::template get<AllocatedSparseCsrTensorType>(
        ctx, place, type);
  }

  pir::Type prim_type();

  const phi::Place &place() const;

  pir::Type dtype() const;

  const phi::DDim &dims() const;

  phi::DataLayout data_layout() const;

  pir::DenseTensorType non_zero_crows() const;

  pir::DenseTensorType non_zero_cols() const;

  pir::DenseTensorType non_zero_elements() const;
};

class AllocatedDenseTensorArrayType
    : public pir::Type::TypeBase<AllocatedDenseTensorArrayType,
                                 pir::Type,
                                 AllocatedDenseTensorArrayTypeStorage,
                                 pir::WrapTypeInterface> {
 public:
  using Base::Base;

  static AllocatedDenseTensorArrayType get(pir::IrContext *ctx,
                                           const phi::Place &place,
                                           dialect::DenseTensorArrayType type) {
    return pir::TypeManager::template get<AllocatedDenseTensorArrayType>(
        ctx, place, type);
  }

  static AllocatedDenseTensorArrayType get(pir::IrContext *ctx,
                                           const phi::Place &place,
                                           const pir::Type &dtype,
                                           const phi::DDim &dims,
                                           const phi::DataLayout &layout) {
    dialect::DenseTensorArrayType type =
        dialect::DenseTensorArrayType::get(ctx, dtype, dims, layout);

    return pir::TypeManager::template get<AllocatedDenseTensorArrayType>(
        ctx, place, type);
  }

  pir::Type prim_type();

  const phi::Place &place() const;

  const pir::Type &dtype() const;

  const pir::DDim &dims() const;

  const phi::DataLayout &data_layout() const;
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedDenseTensorType)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedSelectedRowsType)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedSparseCooTensorType)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedSparseCsrTensorType)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedDenseTensorArrayType)
