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

#include "paddle/fluid/ir/dialect/pd_type_storage.h"
#include "paddle/ir/core/builtin_type_interfaces.h"
#include "paddle/ir/core/type.h"

namespace paddle {
namespace dialect {
///
/// \brief Define built-in parametric types.
///
class DenseTensorType : public ir::Type,  // Not inherite any more
                        public ir::Type::TypeBase<DenseTensorType,
                                                  DenseTensorTypeStorage,
                                                  ir::ShapedTypeInterface> {
 public:
  using Base::Base;
  using Type::Type;

  DECLARE_TYPE_UTILITY_FUNCTOR(DenseTensorType, DenseTensorTypeStorage);

  const ir::Type &dtype() const;

  const phi::DDim &dims() const;

  const phi::DataLayout &data_layout() const;

  const phi::LoD &lod() const;

  const size_t &offset() const;
};

class SelectedRowsType : public ir::Type,  // Not inherite any more
                         public ir::Type::TypeBase<SelectedRowsType,
                                                   SelectedRowsTypeStorage,
                                                   ir::ShapedTypeInterface> {
 public:
  using Base::Base;
  using Type::Type;

  DECLARE_TYPE_UTILITY_FUNCTOR(SelectedRowsType, SelectedRowsTypeStorage);

  const ir::Type &dtype() const;

  const phi::DDim &dims() const;

  const phi::DataLayout &data_layout() const;

  const phi::LoD &lod() const;

  const size_t &offset() const;
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::DenseTensorType)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::SelectedRowsType)
