
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

#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/core/type_base.h"

namespace pir {

class IR_API ContainerType : public Type {
 public:
  using Type::Type;
  static bool classof(Type);
};

class IR_API StackType
    : public Type::TypeBase<StackType, ContainerType, TypeStorage> {
 public:
  using Base::Base;
};

class IR_API InletType : public Type::TypeBase<InletType, Type, TypeStorage> {
 public:
  using Base::Base;
};

class IR_API OutletType : public Type::TypeBase<OutletType, Type, TypeStorage> {
 public:
  using Base::Base;
};

}  // namespace pir

IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::ContainerType)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::StackType)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::InletType)
IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(pir::OutletType)
