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

  const pir::Type &dtype() const;

  const phi::DDim &dims() const;

  const phi::DataLayout &data_layout() const;

  const phi::LoD &lod() const;

  const size_t &offset() const;

  ///
  /// \brief Implementation of 'classof' that compares the type id of
  /// the provided value with the concrete type id.
  ///
  static bool classof(Type type);

  static SelectedRowsType dyn_cast_impl(Type type);
};

class DenseTensorArrayType
    : public pir::Type::TypeBase<DenseTensorArrayType,
                                 pir::Type,
                                 DenseTensorArrayTypeStorage> {
 public:
  using Base::Base;

  const pir::Type &dtype() const;

  const phi::DDim &dims() const;

  const phi::DataLayout &data_layout() const;

  ///
  /// \brief Implementation of 'classof' that compares the type id of
  /// the provided value with the concrete type id.
  ///
  static bool classof(Type type);

  static DenseTensorArrayType dyn_cast_impl(Type type);
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::SelectedRowsType)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::DenseTensorArrayType)
