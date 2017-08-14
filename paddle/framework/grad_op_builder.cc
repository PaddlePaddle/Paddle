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

using VarNameMap = OperatorBase::VarNameMap;

static VarNameMap TransOpArg(const OperatorBase* src_op,
                             const OpArgType& src_type,
                             const OpArgType& dst_type, bool is_grad) {
  const auto& src_inout =
      src_type == OpArgType::IN ? src_op->Inputs() : src_op->Outputs();
  VarNameMap dst_inout;

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
  std::string grad_op_type = OpRegistry::grad_ops().at(op->Type());
  auto I = TransOpArg(op, OpArgType::IN, OpArgType::IN, false);   // I
  auto O = TransOpArg(op, OpArgType::OUT, OpArgType::IN, false);  // O
  auto OG = TransOpArg(op, OpArgType::OUT, OpArgType::IN, true);  // OG
  auto IG = TransOpArg(op, OpArgType::IN, OpArgType::OUT, true);  // IG
  // TODO(merge I/O/OG)
  VarNameMap GradIn;
  GradIn.insert(I.begin(), I.end());
  GradIn.insert(O.begin(), O.end());
  GradIn.insert(OG.begin(), OG.end());

  OperatorBase* grad_op = OpRegistry::op_creators().at(grad_op_type)(
      grad_op_type, GradIn, IG, op->Attrs());
  return grad_op;
}

}  // namespace framework
}  // namespace paddle
