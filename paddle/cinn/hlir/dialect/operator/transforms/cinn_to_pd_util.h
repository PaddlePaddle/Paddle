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
#include <string>
#include <unordered_map>
#include "paddle/cinn/common/macros.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/enforce.h"

namespace pir {
class Block;
class Operation;
}  // namespace pir

namespace cinn::dialect {

using TRule = std::function<::pir::Operation*(::pir::Operation*)>;

class TransformContext {
 private:
  TransformContext();
  DISABLE_COPY_AND_ASSIGN(TransformContext);
  std::unordered_map<std::string, TRule> op_transformers;

 public:
  static TransformContext& Instance() {
    static thread_local TransformContext instance;
    return instance;
  }
  static void Register(std::string cinn_op, TRule rule) {
    Instance().op_transformers[cinn_op] = rule;
  }
  TRule& operator[](std::string cinn_op_name) {
    PADDLE_ENFORCE_NE(
        op_transformers.find(cinn_op_name),
        op_transformers.end(),
        paddle::platform::errors::PreconditionNotMet(
            "op %s has no corresponding transform rules", cinn_op_name));

    return op_transformers[cinn_op_name];
  }
};
void RewriteCinnOpToPdOp(::pir::Block* src_block, ::pir::Block* target_block);
::pir::Operation* RewriteCinnOpToPdOp(::pir::Operation*);

}  // namespace cinn::dialect
