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

#include "paddle/fluid/ir/dialect/kernel_type_storage.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/ir/core/type.h"

namespace paddle {
namespace dialect {

class AllocatedDenseTensorType : public ir::Type {
 public:
  using Type::Type;

  DECLARE_TYPE_UTILITY_FUNCTOR(AllocatedDenseTensorType,
                               AllocatedDenseTensorTypeStorage);

  static AllocatedDenseTensorType get(ir::IrContext *ctx,
                                      const phi::Place &place,
                                      dialect::DenseTensorType type) {
    return ir::TypeManager::template get<AllocatedDenseTensorType>(
        ctx, place, type);
  }

  static AllocatedDenseTensorType get(ir::IrContext *ctx,
                                      const phi::Place &place,
                                      const ir::Type &dtype,
                                      const phi::DDim &dims,
                                      const phi::DataLayout &layout,
                                      const phi::LoD &lod,
                                      size_t offset) {
    dialect::DenseTensorType dense_tensor_type =
        dialect::DenseTensorType::get(ctx, dtype, dims, layout, lod, offset);

    return ir::TypeManager::template get<AllocatedDenseTensorType>(
        ctx, place, dense_tensor_type);
  }

  const phi::Place &place() const;

  const ir::Type &dtype() const;

  const phi::DDim &dims() const;

  const phi::DataLayout &data_layout() const;

  const phi::LoD &lod() const;

  const size_t &offset() const;
};

class AllocatedSelectedRowsType : public ir::Type {
 public:
  using Type::Type;

  DECLARE_TYPE_UTILITY_FUNCTOR(AllocatedSelectedRowsType,
                               AllocatedSelectedRowsTypeStorage);

  static AllocatedSelectedRowsType get(ir::IrContext *ctx,
                                       const phi::Place &place,
                                       dialect::SelectedRowsType type) {
    return ir::TypeManager::template get<AllocatedSelectedRowsType>(
        ctx, place, type);
  }

  static AllocatedSelectedRowsType get(ir::IrContext *ctx,
                                       const phi::Place &place,
                                       const ir::Type &dtype,
                                       const phi::DDim &dims,
                                       const phi::DataLayout &layout,
                                       const phi::LoD &lod,
                                       size_t offset) {
    dialect::SelectedRowsType type =
        dialect::SelectedRowsType::get(ctx, dtype, dims, layout, lod, offset);

    return ir::TypeManager::template get<AllocatedSelectedRowsType>(
        ctx, place, type);
  }

  const phi::Place &place() const;

  const ir::Type &dtype() const;

  const phi::DDim &dims() const;

  const phi::DataLayout &data_layout() const;

  const phi::LoD &lod() const;

  const size_t &offset() const;
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedDenseTensorType)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::AllocatedSelectedRowsType)
