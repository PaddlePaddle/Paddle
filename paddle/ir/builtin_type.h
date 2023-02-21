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
/// \brief Interfaces for user-created built-in types. For example:
/// Type fp32 = Float32Type::get(ctx);
///
class Float32Type : public ir::Type {
 public:
  using Type::Type;

  REGISTER_TYPE_UTILS(Float32Type, ir::TypeStorage);

  static Float32Type get(ir::IrContext *context);
};

class Int32Type : public ir::Type {
 public:
  using Type::Type;

  REGISTER_TYPE_UTILS(Int32Type, ir::TypeStorage);

  static Int32Type get(ir::IrContext *context);
};

}  // namespace ir
