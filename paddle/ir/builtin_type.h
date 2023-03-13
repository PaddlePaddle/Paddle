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

#include "paddle/ir/type.h"

namespace ir {
///
/// \brief This macro is used to get a list of all built-in types in this file.
///
#define GET_BUILT_IN_TYPE_LIST ir::Float32Type, ir::Int32Type

///
/// \brief Definitions of built-in type classes. The built-in type object get
/// method is as follows:
/// \code{cpp}
///   ir::IrContext *ctx = ir::IrContext::Instance();
///   Type fp32 = Float32Type::get(ctx);
/// \endcode
///
class Float32Type : public ir::Type {
 public:
  using Type::Type;

  DECLARE_TYPE_UTILITY_FUNCTOR(Float32Type, ir::TypeStorage);

  static Float32Type get(ir::IrContext *context);
};

class Int32Type : public ir::Type {
 public:
  using Type::Type;

  DECLARE_TYPE_UTILITY_FUNCTOR(Int32Type, ir::TypeStorage);

  static Int32Type get(ir::IrContext *context);
};

}  // namespace ir
