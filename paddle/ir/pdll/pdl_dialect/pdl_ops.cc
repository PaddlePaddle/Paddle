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

#include "paddle/ir/pdll/pdl_dialect/pdl_ops.h"

#include <cstdint>
#include <optional>

#include "paddle/ir/core/attribute.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/enforce.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/type.h"
#include "paddle/ir/core/value.h"
#include "paddle/ir/pdll/pdl_dialect/pdl_types.h"

namespace ir {

namespace pdl {

const char *PDL_PatternOp::attributes_name[2] = {"benefit", "pattern_name"};

void PDL_PatternOp::Build(ir::Builder &builder,
                          ir::OperationArgument &argument,
                          uint32_t benefit,
                          std::optional<std::string> pat_name) {
  ir::Attribute attr_benefit =
      ir::Int32Attribute::get(ir::IrContext::Instance(), benefit);
  argument.AddAttribute("benefit", attr_benefit);
  if (pat_name) {
    ir::Attribute attr_pat_name =
        ir::StrAttribute::get(ir::IrContext::Instance(), pat_name.value());
    argument.AddAttribute("pattern_name", attr_pat_name);
  }

  argument.AddRegion()->emplace_back();
}

Block *PDL_PatternOp::block() {
  IR_ENFORCE(operation());
  IR_ENFORCE(operation()->num_regions() == 1);
  IR_ENFORCE(operation()->region(0).size() == 1);
  return operation()->region(0).front();
}

void PDL_OperandOp::Build(ir::Builder &builder,
                          ir::OperationArgument &argument,
                          OpResult val_of_typeop,
                          Type res) {
  if (val_of_typeop) argument.AddOperand(val_of_typeop);
  argument.AddOutput(res);
}

void PDL_OperandOp::Build(ir::Builder &builder,
                          ir::OperationArgument &argument) {
  Build(
      builder, argument, OpResult(), ValueType::get(ir::IrContext::Instance()));
}

const char *PDL_TypeOp::attributes_name[1] = {"type"};

void PDL_TypeOp::Build(ir::Builder &builder, ir::OperationArgument &argument) {
  Build(builder,
        argument,
        ir::TypeAttribute(),
        ir::pdl::TypeType::get(ir::IrContext::Instance()));
}

void PDL_TypeOp::Build(ir::Builder &builder,
                       ir::OperationArgument &argument,
                       TypeAttribute constant_type,
                       Type res) {
  if (constant_type) {
    argument.AddAttribute("type", constant_type);
  }
  argument.AddOutput(res);
}

const char *PDL_AttributeOp::attributes_name[1] = {"value"};

void PDL_AttributeOp::Build(ir::Builder &builder,
                            ir::OperationArgument &argument,
                            ir::Attribute attr,
                            OpResult val_of_typeop,
                            Type result) {
  if (val_of_typeop) {
    argument.AddOperand(val_of_typeop);
  }
  if (attr) {
    argument.AddAttribute("value", attr);
  }

  argument.AddOutput(result);
}

void PDL_AttributeOp::Build(ir::Builder &builder,
                            ir::OperationArgument &argument) {
  Build(builder,
        argument,
        ir::Attribute(),
        OpResult(),
        AttributeType::get(ir::IrContext::Instance()));
}

void PDL_AttributeOp::Build(ir::Builder &builder,
                            ir::OperationArgument &argument,
                            OpResult val_of_typeop) {
  Build(builder,
        argument,
        ir::Attribute(),
        val_of_typeop,
        AttributeType::get(ir::IrContext::Instance()));
}

void PDL_AttributeOp::Build(ir::Builder &builder,
                            ir::OperationArgument &argument,
                            ir::Attribute attr) {
  Build(builder,
        argument,
        attr,
        OpResult(),
        AttributeType::get(ir::IrContext::Instance()));
}

const char *PDL_OperationOp::attributes_name[2] = {"op_name", "attr_names"};

void PDL_OperationOp::Build(ir::Builder &builder,
                            ir::OperationArgument &argument,
                            const std::string &op_name,
                            const std::vector<std::string> &attr_names,
                            const std::vector<OpResult> &operand_values,
                            const std::vector<OpResult> &attr_values,
                            const std::vector<OpResult> &result_types,
                            Type result) {
  std::vector<ir::OpResult> argument_inputs;
  for (auto v : operand_values) {
    argument_inputs.push_back(v);
  }
  for (auto v : attr_values) {
    argument_inputs.push_back(v);
  }
  for (auto v : result_types) {
    argument_inputs.push_back(v);
  }
  argument.AddOperands(argument_inputs.begin(), argument_inputs.end());

  ir::Attribute attr_op_name =
      ir::StrAttribute::get(ir::IrContext::Instance(), op_name);
  argument.AddAttribute("op_name", attr_op_name);

  std::vector<ir::Attribute> attrs;
  for (size_t i = 0; i < attr_names.size(); ++i) {
    ir::Attribute attr =
        ir::StrAttribute::get(ir::IrContext::Instance(), attr_names[i]);
    attrs.push_back(attr);
  }
  ir::ArrayAttribute array_attr =
      ir::ArrayAttribute::get(ir::IrContext::Instance(), attrs);
  argument.AddAttribute("attr_names", array_attr);

  argument.AddOutput(result);
}

void PDL_EraseOp::Build(ir::Builder &builder,
                        ir::OperationArgument &argument,
                        OpResult val) {
  argument.AddOperand(val);
}

const char *PDL_ResultOp::attributes_name[1] = {"index"};

void PDL_ResultOp::Build(ir::Builder &builder,
                         ir::OperationArgument &argument,
                         uint32_t index,
                         OpResult parent,
                         Type result) {
  argument.AddAttribute(attributes_name[0],
                        Int32Attribute::get(ir::IrContext::Instance(), index));
  argument.AddOperand(parent);
  argument.AddOutput(result);
}

void PDL_ReplaceOp::Build(ir::Builder &builder,
                          ir::OperationArgument &argument,
                          OpResult op_value,
                          OpResult repl_operation,
                          const std::vector<OpResult> &repl_values) {
  IR_ENFORCE(op_value);
  argument.AddOperand(op_value);

  if (repl_operation) {
    argument.AddOperand(repl_operation);
  } else {
    IR_ENFORCE(!repl_values.empty());
    argument.AddOperands(repl_values.begin(), repl_values.end());
  }
}

void PDL_ReplaceOp::Build(ir::Builder &builder,
                          ir::OperationArgument &argument,
                          OpResult op_value,
                          OpResult repl_operation) {
  Build(builder, argument, op_value, repl_operation, {});
}
void PDL_ReplaceOp::Build(ir::Builder &builder,
                          ir::OperationArgument &argument,
                          OpResult op_value,
                          const std::vector<OpResult> &repl_values) {
  Build(builder, argument, op_value, OpResult(), repl_values);
}

const char *PDL_ApplyNativeConstraintOp::attributes_name[1] = {
    "external_function"};

void PDL_ApplyNativeConstraintOp::Build(ir::Builder &builder,
                                        ir::OperationArgument &argument,
                                        const std::string &name,
                                        const std::vector<OpResult> &args) {
  argument.AddAttribute(attributes_name[0],
                        StrAttribute::get(ir::IrContext::Instance(), name));
  argument.AddOperands(args.begin(), args.end());
}

const char *PDL_ApplyNativeRewriteOp::attributes_name[1] = {
    "external_function"};

void PDL_ApplyNativeRewriteOp::Build(ir::Builder &builder,
                                     ir::OperationArgument &argument,
                                     const std::string &name,
                                     const std::vector<OpResult> &args,
                                     const std::vector<Type> &results) {
  argument.AddAttribute(attributes_name[0],
                        StrAttribute::get(ir::IrContext::Instance(), name));
  argument.AddOperands(args.begin(), args.end());
  argument.AddOutputs(results.begin(), results.end());
}

const char *PDL_RewriteOp::attributes_name[1] = {"external_function"};

void PDL_RewriteOp::Build(ir::Builder &builder,
                          ir::OperationArgument &argument,
                          OpResult root) {
  argument.AddOperand(root);

  argument.AddRegion()->emplace_back();
}

void PDL_RewriteOp::Build(ir::Builder &builder,
                          ir::OperationArgument &argument,
                          OpResult root,
                          const std::string &name,
                          const std::vector<OpResult> &args) {
  argument.AddOperand(root);
  argument.AddOperands(args.begin(), args.end());
  argument.AddAttribute(attributes_name[0],
                        StrAttribute::get(ir::IrContext::Instance(), name));

  argument.AddRegion()->emplace_back();
}

Block *PDL_RewriteOp::block() {
  IR_ENFORCE(operation());
  IR_ENFORCE(operation()->num_regions() == 1);
  IR_ENFORCE(operation()->region(0).size() == 1);
  return operation()->region(0).front();
}

}  // namespace pdl
}  // namespace ir

IR_DEFINE_EXPLICIT_TYPE_ID(ir::pdl::PDL_PatternOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::pdl::PDL_TypeOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::pdl::PDL_OperandOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::pdl::PDL_AttributeOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::pdl::PDL_OperationOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::pdl::PDL_EraseOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::pdl::PDL_ResultOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::pdl::PDL_ReplaceOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::pdl::PDL_ApplyNativeConstraintOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::pdl::PDL_ApplyNativeRewriteOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::pdl::PDL_RewriteOp)
