
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

#include "paddle/pir/include/core/sparse_type_storage.h"
#include "paddle/pir/include/core/type.h"
namespace paddle {
namespace dialect {
///
/// \brief Define built-in parameterless types.
///
/// NOTE(zhangbo9674): If you need to directly
/// cache the object of this built-in type in IrContext, please overload the get
/// method, and construct and cache the object in IrContext. For the specific
/// implementation method, please refer to Float16Type.
///
/// The built-in type object get method is as follows:
/// \code{cpp}
///   pir::IrContext *ctx = pir::IrContext::Instance();
///   Type fp32 = Float32Type::get(ctx);
/// \endcode
///

// NOTE(dev): Currently Int8 are not considered as a cached member
// in IrContextImpl because it is not widely used.
class IR_API SparseCooTensorType
    : public pir::Type::
          TypeBase<SparseCooTensorType, pir::Type, SparseCooTensorTypeStorage> {
 public:
  using Base::Base;
  using Type = pir::Type;
  using Dim = SparseCooTensorTypeStorage::Dim;
  using DataLayout = pir::DataLayout;
  using DenseTensorType = pir::DenseTensorType;

  Type dtype() const;
  const Dim &dims() const;
  DataLayout data_layout() const;
  DenseTensorType get_indices() const;
  DenseTensorType get_elements() const;
  bool get_coalesced() const;

  ///
  /// \brief Implementation of 'classof' that compares the type id of
  /// the provided value with the concrete type id.
  ///
  static bool classof(Type type);

  static SparseCooTensorType dyn_cast_impl(Type type);

  static SparseCooTensorType get(pir::IrContext *ctx,
                                 Type dtype,
                                 const Dim &dims,
                                 DataLayout layout,
                                 DenseTensorType non_zero_indices,
                                 DenseTensorType non_zero_elements,
                                 bool coalesced = false) {
    return Base::get(ctx,
                     dtype,
                     dims,
                     layout,
                     non_zero_indices,
                     non_zero_elements,
                     coalesced);
  }
};
}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::SparseCooTensorType)
