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

#include <tuple>
#include <unordered_map>

#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/translator/program_translator.h"
#include "paddle/ir/builtin_type.h"
#include "paddle/ir/dialect.h"
#include "paddle/ir/ir_context.h"

namespace paddle {
namespace translator {

using OpDesc = paddle::framework::OpDesc;
using BlockDesc = paddle::framework::BlockDesc;
using VarDesc = paddle::framework::VarDesc;
using VarType = paddle::framework::proto::VarType;
using TypeTranslateFn = std::function<ir::Type(ir::IrContext*, const VarDesc&)>;

class TypeTranslator {
 private:
  TypeTranslator();  // Disallow instantiation outside of the class.
  std::unordered_map<VarType::Type, TypeTranslateFn> handlers;

 public:
  TypeTranslator(const TypeTranslator&) = delete;
  TypeTranslator& operator=(const TypeTranslator&) = delete;
  TypeTranslator(TypeTranslator&&) = delete;
  TypeTranslator& operator=(TypeTranslator&&) = delete;

  static auto& instance() {
    static TypeTranslator TypeTranslator;
    return TypeTranslator;
  }

  TypeTranslateFn& operator[](VarType::Type type) {
    PADDLE_ENFORCE_NE(
        handlers.count(type),
        0,
        platform::errors::PreconditionNotMet(
            "ProtoType %d has no corresponding translator", type));

    return handlers[type];
  }
};

}  // namespace translator
}  // namespace paddle
