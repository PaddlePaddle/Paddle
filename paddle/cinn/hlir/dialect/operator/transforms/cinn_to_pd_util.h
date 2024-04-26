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
#include "glog/logging.h"
#include "paddle/cinn/common/macros.h"
#include "paddle/common/enforce.h"

namespace pir {
class Block;
class Operation;
class Builder;
}  // namespace pir

namespace cinn::dialect::details {

using TRule =
    std::function<::pir::Operation*(::pir::Operation*, const ::pir::Builder&)>;

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
  static void Register(const std::string cinn_op, TRule rule) {
    VLOG(8) << "Registering transform rule for op " << cinn_op;
    PADDLE_ENFORCE_EQ(
        Instance().op_transformers.find(cinn_op),
        Instance().op_transformers.end(),
        ::common::errors::PreconditionNotMet(
            "op %s 's transform rule already registered", cinn_op));
    Instance().op_transformers.insert({cinn_op, rule});
  }
  TRule& operator[](const std::string cinn_op_name) {
    PADDLE_ENFORCE_NE(
        op_transformers.find(cinn_op_name),
        op_transformers.end(),
        ::common::errors::PreconditionNotMet(
            "op %s has no corresponding transform rules", cinn_op_name));
    VLOG(8) << "Transform rule found for op " << cinn_op_name;
    return op_transformers[cinn_op_name];
  }
};
void RewriteCinnOpToPdOp(const ::pir::Block& src_block,
                         ::pir::Block* target_block);
::pir::Operation* RewriteCinnOpToPdOp(::pir::Operation*, const ::pir::Builder&);

}  // namespace cinn::dialect::details
