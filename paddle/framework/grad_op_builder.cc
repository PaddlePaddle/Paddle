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

static void TransOpArg(const OperatorBase* src_op, const OpArgType& src_type,
                       bool is_grad, OperatorBase::VarNameMap* vars) {
  const auto& src_inout =
      src_type == OpArgType::IN ? src_op->Inputs() : src_op->Outputs();
  auto& dst_inout = *vars;
  const OpProto* proto = OpRegistry::op_info_map().at(src_op->Type()).proto_;
  const auto& src_arg_list =
      src_type == OpArgType::IN ? proto->inputs() : proto->outputs();
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
  auto it = OpRegistry::op_info_map().find(op->Type());
  PADDLE_ENFORCE(it != OpRegistry::op_info_map().end(),
                 "'%s' has not been registered.", op->Type());
  PADDLE_ENFORCE(it->second.proto_ != nullptr, "'%s' has no OpProto.",
                 op->Type());
  std::string grad_op_type = it->second.grad_op_type_;
  PADDLE_ENFORCE(!grad_op_type.empty(), "'%s' has no gradient operator.",
                 op->Type());

  OperatorBase::VarNameMap inputs;
  OperatorBase::VarNameMap outputs;
  TransOpArg(op, OpArgType::IN, false, &inputs);   // I
  TransOpArg(op, OpArgType::OUT, false, &inputs);  // O
  TransOpArg(op, OpArgType::OUT, true, &inputs);   // OG
  TransOpArg(op, OpArgType::IN, true, &outputs);   // IG

  it = OpRegistry::op_info_map().find(grad_op_type);
  PADDLE_ENFORCE(it != OpRegistry::op_info_map().end(),
                 "'%s' has not been registered.", grad_op_type);
  return it->second.creator_(grad_op_type, inputs, outputs, op->Attrs());
}

}  // namespace framework
}  // namespace paddle
