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
                       bool is_grad, VariableNameMap* vars) {
  const auto& src_inout =
      src_type == OpArgType::IN ? src_op->Inputs() : src_op->Outputs();
  auto& dst_inout = *vars;
  auto& proto = OpInfoMap::Instance().Get(src_op->Type()).Proto();
  const auto& src_arg_list =
      src_type == OpArgType::IN ? proto.inputs() : proto.outputs();
  for (const auto& arg : src_arg_list) {
    if (arg.not_in_gradient() && !is_grad) continue;
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
  auto& info = OpInfoMap::Instance().Get(op->Type());
  PADDLE_ENFORCE(info.HasGradientOp());

  VariableNameMap inputs;
  VariableNameMap outputs;
  TransOpArg(op, OpArgType::IN, false, &inputs);   // I
  TransOpArg(op, OpArgType::OUT, false, &inputs);  // O
  TransOpArg(op, OpArgType::OUT, true, &inputs);   // OG
  TransOpArg(op, OpArgType::IN, true, &outputs);   // IG

  auto& grad_info = OpInfoMap::Instance().Get(info.grad_op_type_);
  return grad_info.Creator()(info.grad_op_type_, inputs, outputs, op->Attrs());
}

}  // namespace framework
}  // namespace paddle
