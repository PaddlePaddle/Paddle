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

#include <cstdint>
#include <optional>
#include <string>
#include "paddle/fluid/ir/interface/op_yaml_info.h"
#include "paddle/ir/core/attribute.h"
#include "paddle/ir/core/block.h"
#include "paddle/ir/core/builder.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/op_base.h"
#include "paddle/ir/core/operation_utils.h"
#include "paddle/ir/core/type.h"
#include "paddle/ir/core/value.h"
namespace ir {
namespace pdl {

class PDL_PatternOp : public ir::Op<PDL_PatternOp> {
 public:
  using Op::Op;
  static const char *name() { return "pdl.pattern"; }
  static const char *attributes_name[2];
  static constexpr uint32_t attributes_num = 2;

  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    uint32_t benefit = 1,
                    std::optional<std::string> pat_name = std::nullopt);

  ir::Block *block();
  void Verify() {}
};

class PDL_TypeOp : public ir::Op<PDL_TypeOp> {
 public:
  using Op::Op;
  static const char *name() { return "pdl.type"; }
  static const char *attributes_name[1];
  static constexpr uint32_t attributes_num = 1;

  static void Build(ir::Builder &builder,              // NOLINT
                    ir::OperationArgument &argument);  // NOLINT
  static void Build(ir::Builder &builder,              // NOLINT
                    ir::OperationArgument &argument,   // NOLINT
                    TypeAttribute constant_type /*optional*/,
                    Type res);

  void Verify() {}
};

class PDL_OperandOp : public ir::Op<PDL_OperandOp> {
 public:
  using Op::Op;
  static const char *name() { return "pdl.operand"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;

  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    OpResult val_of_typeop /*optional*/,
                    Type res);
  static void Build(ir::Builder &builder,              // NOLINT
                    ir::OperationArgument &argument);  // NOLINT
  void Verify() {}
  // ir::OpOperand value_type() { return operation()->GetOperandByIndex(0); }
  // ir::OpResult value() { return operation()->GetResultByIndex(0); }
};

class PDL_AttributeOp : public ir::Op<PDL_AttributeOp> {
 public:
  using Op::Op;
  static const char *name() { return "pdl.attribute"; }
  static const char *attributes_name[1];
  static constexpr uint32_t attributes_num = 1;

  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    ir::Attribute attr,
                    OpResult val_of_typeop,
                    Type result);
  static void Build(ir::Builder &builder,              // NOLINT
                    ir::OperationArgument &argument);  // NOLINT
  static void Build(ir::Builder &builder,              // NOLINT
                    ir::OperationArgument &argument,   // NOLINT
                    OpResult val_of_typeop);
  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    ir::Attribute attr);

  void Verify() {}
};

class PDL_OperationOp : public ir::Op<PDL_OperationOp> {
 public:
  using Op::Op;
  static const char *name() { return "pdl.operation"; }
  static const char *attributes_name[2];
  static constexpr uint32_t attributes_num = 2;

  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    const std::string &op_name,
                    const std::vector<std::string> &attr_names,
                    const std::vector<OpResult> &operand_values,
                    const std::vector<OpResult> &attr_values,
                    const std::vector<OpResult> &result_types,
                    Type result);

  void Verify() {}
};

class PDL_EraseOp : public ir::Op<PDL_EraseOp> {
 public:
  using Op::Op;
  static const char *name() { return "pdl.erase"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;

  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    OpResult val);

  void Verify() {}
};

class PDL_ResultOp : public ir::Op<PDL_ResultOp> {
 public:
  using Op::Op;
  static const char *name() { return "pdl.result"; }
  static const char *attributes_name[1];
  static constexpr uint32_t attributes_num = 1;

  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    uint32_t index,
                    OpResult parent,
                    Type out);

  void Verify() {}
};

class PDL_ReplaceOp : public ir::Op<PDL_ReplaceOp> {
 public:
  using Op::Op;
  static const char *name() { return "pdl.replace"; }
  static constexpr const char **attributes_name = nullptr;
  static constexpr uint32_t attributes_num = 0;

  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    OpResult op_value,
                    OpResult repl_operation,
                    const std::vector<OpResult> &repl_values);
  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    OpResult op_value,
                    OpResult repl_operation);
  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    OpResult op_value,
                    const std::vector<OpResult> &repl_values);

  void Verify() {}
};

class PDL_ApplyNativeConstraintOp : public ir::Op<PDL_ApplyNativeConstraintOp> {
 public:
  using Op::Op;
  static const char *name() { return "pdl.apply_native_constraint"; }
  static const char *attributes_name[1];
  static constexpr uint32_t attributes_num = 1;

  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    const std::string &name,
                    const std::vector<OpResult> &args);

  void Verify() {}
};

class PDL_ApplyNativeRewriteOp : public ir::Op<PDL_ApplyNativeRewriteOp> {
 public:
  using Op::Op;
  static const char *name() { return "pdl.apply_native_rewrite"; }
  static const char *attributes_name[1];
  static constexpr uint32_t attributes_num = 1;

  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    const std::string &name,
                    const std::vector<OpResult> &args,
                    const std::vector<Type> &results);

  void Verify() {}
};

class PDL_RewriteOp : public ir::Op<PDL_RewriteOp> {
 public:
  using Op::Op;
  static const char *name() { return "pdl.rewrite"; }
  static const char *attributes_name[1];
  static constexpr uint32_t attributes_num = 1;

  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    OpResult root,
                    const std::string &name,
                    const std::vector<OpResult> &args);

  static void Build(ir::Builder &builder,             // NOLINT
                    ir::OperationArgument &argument,  // NOLINT
                    OpResult root);

  Block *block();
  void Verify() {}
};

}  // namespace pdl
}  // namespace ir

IR_DECLARE_EXPLICIT_TYPE_ID(ir::pdl::PDL_PatternOp)
IR_DECLARE_EXPLICIT_TYPE_ID(ir::pdl::PDL_TypeOp)
IR_DECLARE_EXPLICIT_TYPE_ID(ir::pdl::PDL_OperandOp)
IR_DECLARE_EXPLICIT_TYPE_ID(ir::pdl::PDL_AttributeOp)
IR_DECLARE_EXPLICIT_TYPE_ID(ir::pdl::PDL_OperationOp)
IR_DECLARE_EXPLICIT_TYPE_ID(ir::pdl::PDL_EraseOp)
IR_DECLARE_EXPLICIT_TYPE_ID(ir::pdl::PDL_ResultOp)
IR_DECLARE_EXPLICIT_TYPE_ID(ir::pdl::PDL_ReplaceOp)
IR_DECLARE_EXPLICIT_TYPE_ID(ir::pdl::PDL_ApplyNativeConstraintOp)
IR_DECLARE_EXPLICIT_TYPE_ID(ir::pdl::PDL_ApplyNativeRewriteOp)
IR_DECLARE_EXPLICIT_TYPE_ID(ir::pdl::PDL_RewriteOp)
