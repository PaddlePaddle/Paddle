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

typedef std::vector<int> Ints;

enum class OpArgType { IN, OUT };

const Ints* AttrFormat(const AttributeMap& attrs, const std::string& key) {
  return (attrs.count(key) > 0) ? &boost::get<Ints>(attrs.at(key)) : nullptr;
}

Ints* AttrFormat(AttributeMap& attrs, const std::string& key) {
  return (attrs.count(key) > 0) ? &boost::get<Ints>(attrs.at(key)) : nullptr;
}

static void TransOpArg(const OperatorBase* src_op,
                       std::vector<std::string>& grad_inputs,
                       std::vector<std::string>& grad_outputs,
                       AttributeMap& grad_attrs,
                       std::unordered_map<std::string, int>& grad_idxs,
                       const std::string& src_type, const std::string& dst_type,
                       int& idx, bool is_grad) {
  const std::vector<std::string>& src_inout =
      (src_type == "input_format") ? src_op->inputs_ : src_op->outputs_;

  const std::vector<int>* src_format = AttrFormat(src_op->Attrs(), src_type);

  std::vector<std::string>& dst_inout =
      (dst_type == "input_format") ? grad_inputs : grad_outputs;

  std::vector<int>* dst_format = AttrFormat(grad_attrs, dst_type);

  const OpProto& proto = *(OpRegistry::op_info_map().at(src_op->type_).proto_);

  const auto& src_arg_list =
      (src_type == "input_format") ? proto.inputs() : proto.outputs();

  for (const auto& arg : src_arg_list) {
    std::string src_name = arg.name();
    std::string dst_name = is_grad ? src_name + kGradVarSuffix : src_name;
    grad_idxs[dst_name] = idx++;
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
  auto it = OpRegistry::op_info_map().find(op->type_);
  PADDLE_ENFORCE(it != OpRegistry::op_info_map().end(),
                 "'%s' has not been registered.", op->type_);
  std::string grad_op_type = it->second.grad_op_type_;
  PADDLE_ENFORCE(!grad_op_type.empty(), "'%s' has no gradient operator.",
                 op->type_);

  AttributeMap grad_attrs(op->Attrs());
  grad_attrs.erase("input_format");
  grad_attrs.erase("output_format");
  if (op->Attrs().count("input_format") > 0) {
    grad_attrs["output_format"] = std::vector<int>({0});
  }
  if (op->Attrs().count("input_format") > 0 ||
      op->Attrs().count("output_format") > 0) {
    grad_attrs["input_format"] = std::vector<int>({0});
  }

  std::vector<std::string> grad_inputs, grad_outputs;

  using VarIndexMap = std::unordered_map<std::string, int>;
  VarIndexMap* grad_idxs = new VarIndexMap;
  int in_idx = 0;
  int out_idx = 0;
  TransOpArg(op, grad_inputs, grad_outputs, grad_attrs, *grad_idxs,
             "input_format", "input_format", in_idx, false);  // I
  TransOpArg(op, grad_inputs, grad_outputs, grad_attrs, *grad_idxs,
             "output_format", "input_format", in_idx, false);  // G
  TransOpArg(op, grad_inputs, grad_outputs, grad_attrs, *grad_idxs,
             "output_format", "input_format", in_idx, true);  // OG
  TransOpArg(op, grad_inputs, grad_outputs, grad_attrs, *grad_idxs,
             "input_format", "output_format", out_idx, true);  // IG

  it = OpRegistry::op_info_map().find(grad_op_type);
  PADDLE_ENFORCE(it != OpRegistry::op_info_map().end(),
                 "'%s' has not been registered.", grad_op_type);
  OperatorBase* grad_op = it->second.creator_();

  grad_op->type_ = grad_op_type;
  grad_op->inputs_ = grad_inputs;
  grad_op->outputs_ = grad_outputs;
  grad_op->attrs_ = grad_attrs;
  grad_op->in_out_idxs_.reset(grad_idxs);

  return grad_op;
}

}  // namespace framework
}  // namespace paddle
