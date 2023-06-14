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

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/ir_adaptor/translator/program_translator.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/value.h"

namespace paddle {
namespace translator {

class OpTranslator {
 public:
  using ResultIdx = size_t;
  using OpDesc = paddle::framework::OpDesc;
  using BlockDesc = paddle::framework::BlockDesc;
  using VarDesc = paddle::framework::VarDesc;
  using OpTranslateFn = std::function<ir::Operation*(
      ir::IrContext*, TranslationContext*, ir::Program*, const OpDesc&)>;

 private:
  OpTranslator();  // Disallow instantiation outside of the class.
  std::unordered_map<std::string, OpTranslateFn> special_handlers;
  OpTranslateFn general_handler;

 public:
  OpTranslator(const OpTranslator&) = delete;
  OpTranslator& operator=(const OpTranslator&) = delete;
  OpTranslator(OpTranslator&&) = delete;
  OpTranslator& operator=(OpTranslator&&) = delete;

  static auto& instance() {
    static OpTranslator OpTranslator;
    return OpTranslator;
  }

  OpTranslateFn& operator[](const std::string& op_type) {
    if (special_handlers.count(op_type) == 0) {
      return general_handler;
    } else {
      return special_handlers[op_type];
    }
  }
};

using OpTranslateFn = OpTranslator::OpTranslateFn;

}  // namespace translator
}  // namespace paddle
