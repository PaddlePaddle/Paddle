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
class IrMapping;
class Block;
class Operation;
class PatternRewriter;
class IrMapping;
}  // namespace pir

namespace cinn::dialect::details {

using TRule = std::function<::pir::Operation*(
    ::pir::Operation*, ::pir::IrMapping&, ::pir::PatternRewriter&)>;

class TransformContext {
 private:
  TransformContext() = default;
  DISABLE_COPY_AND_ASSIGN(TransformContext);
  std::unordered_map<std::string, TRule> op_transformers;

 public:
  static TransformContext& Instance() {
    static thread_local TransformContext instance;
    return instance;
  }
  void Insert(const std::string op_name, TRule rule) {
    VLOG(8) << "Inserting transform rule for op " << op_name;
    PADDLE_ENFORCE_EQ(
        op_transformers.find(op_name),
        op_transformers.end(),
        ::common::errors::PreconditionNotMet(
            "op %s 's transform rule already registered", op_name));
    op_transformers.insert({op_name, rule});
  }
  TRule& operator[](const std::string op_name) {
    PADDLE_ENFORCE_NE(
        op_transformers.find(op_name),
        op_transformers.end(),
        ::common::errors::PreconditionNotMet(
            "op %s has no corresponding transform rules", op_name));
    VLOG(8) << "Transform rule found for op " << op_name;
    return op_transformers[op_name];
  }
};

class TransformRegistrar {
 public:
  void Touch() {}
  explicit TransformRegistrar(const std::string op_name, TRule rule) {
    TransformContext::Instance().Insert(op_name, rule);
  }
};

#define REGISTER_TRANSFORM_RULES(registrar_name, op_name, rule_handler_name)  \
  STATIC_ASSERT_GLOBAL_NAMESPACE(                                             \
      __reg_op_transform_rules__##registrar_name,                             \
      "REGISTER_TRANSFORM_RULES must be called in global namespace");         \
  static cinn::dialect::details::TransformRegistrar                           \
      __op_transform_rules_registrar_##registrar_name##__(op_name,            \
                                                          rule_handler_name); \
  int TouchOpTransformerRulesRegistrar_##registrar_name() {                   \
    __op_transform_rules_registrar_##registrar_name##__.Touch();              \
    return 0;                                                                 \
  }                                                                           \
  static cinn::dialect::details::TransformRegistrar&                          \
      __op_transform_rules_tmp_registrar_##registrar_name##__ UNUSED =        \
          __op_transform_rules_registrar_##registrar_name##__

void RewriteCinnOpToPdOp(const ::pir::Block& src_block,
                         ::pir::Block* target_block,
                         ::pir::PatternRewriter& rewriter);  // NOLINT
::pir::Operation* RewriteCinnOpToPdOp(::pir::Operation*,
                                      ::pir::IrMapping&,         // NOLINT
                                      ::pir::PatternRewriter&);  // NOLINT

}  // namespace cinn::dialect::details
