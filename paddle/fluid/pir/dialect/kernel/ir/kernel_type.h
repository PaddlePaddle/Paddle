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
#include "paddle/pir/core/type.h"

namespace paddle {
namespace dialect {

class AllocatedDenseTensorType
    : public pir::Type::TypeBase<AllocatedDenseTensorType,
                                 pir::Type,
                                 AllocatedDenseTensorTypeStorage> {
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
                                      const phi::LoD &lod,
                                      size_t offset) {
    dialect::DenseTensorType dense_tensor_type =
        dialect::DenseTensorType::get(ctx, dtype, dims, layout, lod, offset);

    return pir::TypeManager::template get<AllocatedDenseTensorType>(
        ctx, place, dense_tensor_type);
  }

  const phi::Place &place() const;

  const pir::Type &dtype() const;

  const phi::DDim &dims() const;

  const phi::DataLayout &data_layout() const;

  const phi::LoD &lod() const;

  const size_t &offset() const;
};

class AllocatedSelectedRowsType
    : public pir::Type::TypeBase<AllocatedSelectedRowsType,
                                 pir::Type,
                                 AllocatedSelectedRowsTypeStorage> {
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
                                       const phi::LoD &lod,
                                       size_t offset) {
    dialect::SelectedRowsType type =
        dialect::SelectedRowsType::get(ctx, dtype, dims, layout, lod, offset);

    return pir::TypeManager::template get<AllocatedSelectedRowsType>(
        ctx, place, type);
  }

  const phi::Place &place() const;

  const pir::Type &dtype() const;

  const phi::DDim &dims() const;

  const phi::DataLayout &data_layout() const;

  const phi::LoD &lod() const;

  const size_t &offset() const;
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedDenseTensorType)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedSelectedRowsType)
