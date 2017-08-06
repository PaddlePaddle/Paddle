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
#include "paddle/framework/op_proto.pb.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace framework {

class OpRegistry;

using VarIndexMap = std::unordered_map<std::string, int>;

enum class OpArgType { IN, OUT };

static std::vector<int>* GetOpFormat(OperatorBase* op, const OpArgType& type) {
  std::string key = type == OpArgType::IN ? "input_format" : "output_format";
  return op->attrs_.count(key)
             ? &boost::get<std::vector<int>>(op->attrs_.at(key))
             : nullptr;
}

static const std::vector<int>* GetOpFormat(const OperatorBase* op,
                                           const OpArgType& type) {
  std::string key = type == OpArgType::IN ? "input_format" : "output_format";
  return op->attrs_.count(key)
             ? &boost::get<std::vector<int>>(op->attrs_.at(key))
             : nullptr;
}

static void TransOpArg(const OperatorBase* src_op, OperatorBase* dst_op,
                       const OpArgType& src_type, const OpArgType& dst_type,
                       int& idx, bool is_grad) {
  const std::vector<std::string>& src_inout =
      src_type == OpArgType::IN ? src_op->inputs_ : src_op->outputs_;
  const std::vector<int>* src_format = GetOpFormat(src_op, src_type);

  std::vector<std::string>& dst_inout =
      dst_type == OpArgType::IN ? dst_op->inputs_ : dst_op->outputs_;
  std::vector<int>* dst_format = GetOpFormat(dst_op, dst_type);
  const OpProto& proto = OpRegistry::protos().at(src_op->type_);
  const auto& src_arg_list =
      src_type == OpArgType::IN ? proto.inputs() : proto.outputs();

  for (const auto& arg : src_arg_list) {
    std::string src_name = arg.name();
    std::string dst_name = is_grad ? src_name + kGradVarSuffix : src_name;
    (*dst_op->in_out_idxs_)[dst_name] = idx++;
    int src_arg_idx = src_op->in_out_idxs_->at(src_name);
    int src_begin =
        src_format == nullptr ? src_arg_idx : src_format->at(src_arg_idx);
    int src_end = src_format == nullptr ? src_arg_idx + 1
                                        : src_format->at(src_arg_idx + 1);
    for (int i = src_begin; i < src_end; ++i) {
      std::string s =
          is_grad ? src_inout[i] + kGradVarSuffix
                  : (arg.ignore_gradient() ? kEmptyVarName : src_inout[i]);
      dst_inout.emplace_back(s);
    }
    if (dst_format != nullptr) {
      dst_format->push_back(dst_inout.size());
    }
  }
}

OperatorBase* BuildGradOp(const OperatorBase* op) {
  std::string grad_op_type = OpRegistry::grad_ops().at(op->type_);
  OperatorBase* grad_op = OpRegistry::op_creators().at(grad_op_type)();
  grad_op->type_ = grad_op_type;
  grad_op->attrs_ = op->attrs_;
  grad_op->attrs_.erase("input_format");
  grad_op->attrs_.erase("output_format");
  if (GetOpFormat(op, OpArgType::IN) != nullptr) {
    grad_op->attrs_["output_format"] = std::vector<int>({0});
  }
  if (GetOpFormat(op, OpArgType::IN) != nullptr ||
      GetOpFormat(op, OpArgType::OUT) != nullptr) {
    grad_op->attrs_["input_format"] = std::vector<int>({0});
  }
  grad_op->in_out_idxs_.reset(new VarIndexMap());
  int in_idx = 0;
  int out_idx = 0;
  TransOpArg(op, grad_op, OpArgType::IN, OpArgType::IN, in_idx, false);   // I
  TransOpArg(op, grad_op, OpArgType::OUT, OpArgType::IN, in_idx, false);  // G
  TransOpArg(op, grad_op, OpArgType::OUT, OpArgType::IN, in_idx, true);   // OG
  TransOpArg(op, grad_op, OpArgType::IN, OpArgType::OUT, out_idx, true);  // IG
  return grad_op;
}

}  // namespace framework
}  // namespace paddle
