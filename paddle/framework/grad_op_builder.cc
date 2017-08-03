/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/grad_op_builder.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace framework {

OperatorBase* GradOpBuilder::Build() {
  BuildOpInOutArgList();
  std::string grad_op_type = OpRegistry::grad_ops().at(op_.type_);
  OperatorBase* grad_op = OpRegistry::op_creators().at(grad_op_type)();
  grad_op->type_ = grad_op_type;
  CompleteGradOp(grad_op);
  return grad_op;
}

OpInOutArg* GradOpBuilder::BuildArg(const VarProto& var,
                                    const VarIndexMap& var_map,
                                    const std::vector<int>& format,
                                    InOutType type) {
  int idx = var_map.at(var.name());
  int begin_idx = format.empty() ? idx : format.at(idx);
  int end_idx = format.empty() ? idx + 1 : format.at(idx + 1);
  return new OpInOutArg(var.name(), type, !var.ignore_gradient(), begin_idx,
                        end_idx);
}

void GradOpBuilder::BuildOpInOutArgList() {
  const OpProto& op_proto = OpRegistry::protos().at(op_.type_);
  const auto& var_map = *(OpRegistry::VarIndexMaps().at(op_.type_));
  const std::vector<int>& in_format =
      op_.attrs_.count("input_format")
          ? op_.GetAttr<std::vector<int>>("input_format")
          : std::vector<int>();
  const std::vector<int>& out_format =
      op_.attrs_.count("output_format")
          ? op_.GetAttr<std::vector<int>>("output_format")
          : std::vector<int>();
  for (const auto& var : op_proto.inputs()) {
    arg_list_.emplace_back(
        std::shared_ptr<OpInOutArg>(BuildArg(var, var_map, in_format, IN)));
  }
  for (const auto& var : op_proto.outputs()) {
    arg_list_.emplace_back(
        std::shared_ptr<OpInOutArg>(BuildArg(var, var_map, out_format, OUT)));
  }
}

void GradOpBuilder::AddArgIntoGradOp(const OpInOutArg* arg,
                                     std::vector<std::string>& in_out,
                                     std::vector<int>& format,
                                     VarIndexMap* varmap, int& idx,
                                     bool is_grad) const {
  std::string var_name = arg->proto_name_;
  if (is_grad) {
    var_name += OperatorBase::GRAD_VAR_SUFFIX();
  }
  (*varmap)[var_name] = idx++;
  size_t pre_sz = in_out.size();
  auto base_it = arg->type_ == IN ? op_.inputs_.begin() : op_.outputs_.begin();
  std::copy(base_it + arg->begin_idx_, base_it + arg->end_idx_,
            std::back_inserter(in_out));
  if (is_grad) {
    for (size_t i = pre_sz; i < in_out.size(); ++i) {
      in_out[i] += OperatorBase::GRAD_VAR_SUFFIX();
    }
  }
  format.push_back(in_out.size());
}

void GradOpBuilder::CompleteGradOp(OperatorBase* grad_op) const {
  grad_op->attrs_ = op_.attrs_;
  grad_op->attrs_.erase("input_format");
  grad_op->attrs_.erase("output_format");
  VarIndexMap* grad_varmap = new VarIndexMap();
  int in_idx = 0;
  int out_idx = 0;
  std::vector<int> in_format({0});
  std::vector<int> out_format({0});
  for (const auto& arg : arg_list_) {
    // op_'s inputs_ and outputs_
    if (arg->needed_in_grad_) {
      AddArgIntoGradOp(arg.get(), grad_op->inputs_, in_format, grad_varmap,
                       in_idx, false);
    }
    if (arg->type_ == IN) {
      // gradients of op_'s inputs_
      AddArgIntoGradOp(arg.get(), grad_op->outputs_, out_format, grad_varmap,
                       out_idx, true);
    } else {
      // gradients of op_'s outputs_
      AddArgIntoGradOp(arg.get(), grad_op->inputs_, in_format, grad_varmap,
                       in_idx, true);
    }
  }
  grad_op->attrs_["input_format"] = in_format;
  grad_op->attrs_["output_format"] = out_format;
  grad_op->in_out_idxs_.reset(grad_varmap);
}

}  // namespace framework
}  // namespace paddle
