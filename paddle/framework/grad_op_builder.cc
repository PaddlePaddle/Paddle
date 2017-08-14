/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOpArgType::OUT WARRANTIES OR CONDITIONS OF ANY KOpArgType::IND, either
express or implied. See the License for the specific language governing
permissions and limitations under the License. */

#include "paddle/framework/grad_op_builder.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace framework {
enum class OpArgType { IN, OUT };

static void TransOpArg(const OperatorBase* src_op,
                       OperatorBase::VarNameMap* vars,
                       const OpArgType& src_type, bool is_grad) {
  const auto& src_inout =
      src_type == OpArgType::IN ? src_op->inputs_ : src_op->outputs_;
  auto& dst_inout = *vars;

  const OpProto& proto = OpProtos().at(src_op->Type());
  const auto& src_arg_list =
      src_type == OpArgType::IN ? proto.inputs() : proto.outputs();
  for (const auto& arg : src_arg_list) {
    if (arg.no_gradient() && !is_grad) continue;
    const std::string src_name = arg.name();
    std::string dst_name = is_grad ? GradVarName(src_name) : src_name;
    dst_inout[dst_name].reserve(src_inout.at(src_name).size());
    for (auto& var_name : src_inout.at(src_name)) {
      std::string s = is_grad ? GradVarName(var_name) : var_name;
      dst_inout[dst_name].emplace_back(s);
    }
  }
  return dst_inout;
}

OperatorBase* BuildGradOp(const OperatorBase* op) {
  auto gop_type_it = OpRegistry::grad_ops().find(op->type_);
  PADDLE_ENFORCE(gop_type_it != OpRegistry::grad_ops().end(),
                 "Operator %s do not register gradient type", op->type_);
  auto& grad_op_type = gop_type_it->second;
  OperatorBase::VarNameMap inputs;
  OperatorBase::VarNameMap outputs;
  TransOpArg(op, &inputs, OpArgType::IN, false);   // I
  TransOpArg(op, &inputs, OpArgType::OUT, false);  // O
  TransOpArg(op, &inputs, OpArgType::OUT, true);   // OG
  TransOpArg(op, &outputs, OpArgType::IN, true);   // IG
  auto gop_it = OpRegistry::op_creators().find(grad_op_type);
  PADDLE_ENFORCE(gop_it != OpRegistry::op_creators().end(),
                 "Operator %s 's Gradient %s's creator cannot be found",
                 op->type_, grad_op_type);

  return gop_it->second(grad_op_type, inputs, outputs, op->attrs_);
}

}  // namespace framework
}  // namespace paddle
