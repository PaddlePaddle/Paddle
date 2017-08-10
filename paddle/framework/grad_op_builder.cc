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
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace framework {

class OpRegistry;

enum class OpArgType { IN, OUT };

static void TransOpArg(const OperatorBase* src_op, OperatorBase* dst_op,
                       const OpArgType& src_type, const OpArgType& dst_type,
                       bool is_grad) {
  const auto& src_inout =
      src_type == OpArgType::IN ? src_op->inputs_ : src_op->outputs_;
  auto& dst_inout =
      dst_type == OpArgType::IN ? dst_op->inputs_ : dst_op->outputs_;

  const OpProto& proto = OpProtos().at(src_op->type_);
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
}

OperatorBase* BuildGradOp(const OperatorBase* op) {
  std::string grad_op_type = OpRegistry::grad_ops().at(op->type_);
  OperatorBase* grad_op = OpRegistry::op_creators().at(grad_op_type)();
  grad_op->type_ = grad_op_type;
  grad_op->attrs_ = op->attrs_;
  TransOpArg(op, grad_op, OpArgType::IN, OpArgType::IN, false);   // I
  TransOpArg(op, grad_op, OpArgType::OUT, OpArgType::IN, false);  // O
  TransOpArg(op, grad_op, OpArgType::OUT, OpArgType::IN, true);   // OG
  TransOpArg(op, grad_op, OpArgType::IN, OpArgType::OUT, true);   // IG
  return grad_op;
}

}  // namespace framework
}  // namespace paddle
