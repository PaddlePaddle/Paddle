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

#include "paddle/fluid/pir/dialect/operator/ir/type_storage.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/builtin_type_interfaces.h"
#include "paddle/pir/include/core/dll_decl.h"
#include "paddle/pir/include/core/type.h"
#include "paddle/utils/test_macros.h"

namespace paddle {
namespace dialect {

using DenseTensorType = pir::DenseTensorType;

class TEST_API SelectedRowsType
    : public pir::Type::TypeBase<SelectedRowsType,
                                 pir::Type,
                                 SelectedRowsTypeStorage,
                                 pir::ShapedTypeInterface> {
 public:
  using Base::Base;

  static std::string name() { return "t_selected_rows"; }

  const pir::Type &dtype() const;

  const phi::DDim &dims() const;

  const phi::DataLayout &data_layout() const;

  const phi::LegacyLoD &lod() const;

  const size_t &offset() const;

  ///
  /// \brief Implementation of 'classof' that compares the type id of
  /// the provided value with the concrete type id.
  ///
  static bool classof(Type type);

  static SelectedRowsType dyn_cast_impl(Type type);

  static SelectedRowsType get(pir::IrContext *ctx,
                              Type dtype,
                              const phi::DDim &dims,
                              DataLayout layout = DataLayout::kNCHW,
                              const phi::LegacyLoD &lod = {},
                              size_t offset = 0u) {
    return Base::get(ctx, dtype, dims, layout, lod, offset);
  }
};

class IR_API DenseTensorArrayType
    : public pir::Type::TypeBase<DenseTensorArrayType,
                                 pir::Type,
                                 DenseTensorArrayTypeStorage> {
 public:
  using Base::Base;

  static std::string name() { return "t_dense_tensor_array"; }

  const pir::Type &dtype() const;

  const phi::DDim &dims() const;

  const phi::DataLayout &data_layout() const;

  ///
  /// \brief Implementation of 'classof' that compares the type id of
  /// the provided value with the concrete type id.
  ///
  static bool classof(Type type);

  static DenseTensorArrayType dyn_cast_impl(Type type);

  static DenseTensorArrayType get(pir::IrContext *ctx,
                                  Type dtype,
                                  const phi::DDim &dims,
                                  DataLayout layout = DataLayout::kNCHW) {
    return Base::get(ctx, dtype, dims, layout);
  }
};

class IR_API SparseCooTensorType
    : public pir::Type::TypeBase<SparseCooTensorType,
                                 pir::Type,
                                 SparseCooTensorTypeStorage,
                                 pir::ShapedTypeInterface> {
 public:
  using Base::Base;

  static std::string name() { return "t_sparse_coo_tensor"; }

  pir::Type dtype() const;
  const common::DDim &dims() const;
  const common::DDim &non_zero_dims() const;
  common::DataLayout data_layout() const;
  pir::DenseTensorType non_zero_indices() const;
  pir::DenseTensorType non_zero_elements() const;
  bool coalesced() const;

  ///
  /// \brief Implementation of 'classof' that compares the type id of
  /// the provided value with the concrete type id.
  ///
  static bool classof(pir::Type type);

  static SparseCooTensorType dyn_cast_impl(pir::Type type);

  static SparseCooTensorType get(pir::IrContext *ctx,
                                 pir::Type dtype,
                                 const common::DDim &dims,
                                 const common::DDim &non_zero_dims,
                                 common::DataLayout layout,
                                 pir::DenseTensorType non_zero_indices,
                                 pir::DenseTensorType non_zero_elements,
                                 bool coalesced = false) {
    return Base::get(ctx,
                     dtype,
                     dims,
                     non_zero_dims,
                     layout,
                     non_zero_indices,
                     non_zero_elements,
                     coalesced);
  }
};

class IR_API SparseCsrTensorType
    : public pir::Type::TypeBase<SparseCsrTensorType,
                                 pir::Type,
                                 SparseCsrTensorTypeStorage,
                                 pir::ShapedTypeInterface> {
 public:
  using Base::Base;

  static std::string name() { return "t_sparse_csr_tensor"; }

  pir::Type dtype() const;
  const common::DDim &dims() const;
  common::DataLayout data_layout() const;
  pir::DenseTensorType non_zero_crows() const;
  pir::DenseTensorType non_zero_cols() const;
  pir::DenseTensorType non_zero_elements() const;

  ///
  /// \brief Implementation of 'classof' that compares the type id of
  /// the provided value with the concrete type id.
  ///
  static bool classof(pir::Type type);

  static SparseCsrTensorType dyn_cast_impl(pir::Type type);

  static SparseCsrTensorType get(pir::IrContext *ctx,
                                 pir::Type dtype,
                                 const common::DDim &dims,
                                 common::DataLayout layout,
                                 pir::DenseTensorType non_zero_crows,
                                 pir::DenseTensorType non_zero_cols,
                                 pir::DenseTensorType non_zero_elements) {
    return Base::get(ctx,
                     dtype,
                     dims,
                     layout,
                     non_zero_crows,
                     non_zero_cols,
                     non_zero_elements);
  }
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::SelectedRowsType)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::DenseTensorArrayType)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::SparseCooTensorType)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::SparseCsrTensorType)
