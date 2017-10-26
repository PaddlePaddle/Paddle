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

#include "paddle/framework/op_desc.h"
#include <functional>
#include <unordered_map>
#include "paddle/framework/block_desc.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/program_desc.h"

namespace paddle {
namespace framework {

OpDescBind::OpDescBind(const std::string &type, const VariableNameMap &inputs,
                       const VariableNameMap &outputs,
                       const AttributeMap &attrs) {
  desc_.set_type(type);
  inputs_ = inputs;
  outputs_ = outputs;
  attrs_ = attrs;
  need_update_ = true;
}

OpDescBind::OpDescBind(const OpDesc &desc, ProgramDescBind *prog)
    : desc_(desc), need_update_(false) {
  // restore inputs_
  int input_size = desc_.inputs_size();
  for (int i = 0; i < input_size; ++i) {
    const OpDesc::Var &var = desc_.inputs(i);
    std::vector<std::string> &args = inputs_[var.parameter()];
    int argu_size = var.arguments_size();
    args.reserve(argu_size);
    for (int j = 0; j < argu_size; ++j) {
      args.push_back(var.arguments(j));
    }
  }
  // restore outputs_
  int output_size = desc_.outputs_size();
  for (int i = 0; i < output_size; ++i) {
    const OpDesc::Var &var = desc_.outputs(i);
    std::vector<std::string> &args = outputs_[var.parameter()];
    int argu_size = var.arguments_size();
    args.reserve(argu_size);
    for (int j = 0; j < argu_size; ++j) {
      args.push_back(var.arguments(j));
    }
  }
  // restore attrs_
  for (const OpDesc::Attr &attr : desc_.attrs()) {
    std::string attr_name = attr.name();
    attrs_[attr_name] = GetAttrValue(attr, prog->Proto());
  }
}

OpDesc *OpDescBind::Proto() {
  Flush();
  return &desc_;
}

const std::vector<std::string> &OpDescBind::Input(
    const std::string &name) const {
  auto it = inputs_.find(name);
  PADDLE_ENFORCE(it != inputs_.end(), "Input %s cannot be found in Op %s", name,
                 Type());
  return it->second;
}

std::vector<std::string> OpDescBind::InputArgumentNames() const {
  std::vector<std::string> retv;
  for (auto &ipt : this->inputs_) {
    retv.insert(retv.end(), ipt.second.begin(), ipt.second.end());
  }
  return retv;
}

void OpDescBind::SetInput(const std::string &param_name,
                          const std::vector<std::string> &args) {
  need_update_ = true;
  inputs_[param_name] = args;
}

const std::vector<std::string> &OpDescBind::Output(
    const std::string &name) const {
  auto it = outputs_.find(name);
  PADDLE_ENFORCE(it != outputs_.end(), "Output %s cannot be found in Op %s",
                 name, Type());
  return it->second;
}

std::vector<std::string> OpDescBind::OutputArgumentNames() const {
  std::vector<std::string> retv;
  for (auto &ipt : this->outputs_) {
    retv.insert(retv.end(), ipt.second.begin(), ipt.second.end());
  }
  return retv;
}

void OpDescBind::SetOutput(const std::string &param_name,
                           const std::vector<std::string> &args) {
  need_update_ = true;
  this->outputs_[param_name] = args;
}

AttrType OpDescBind::GetAttrType(const std::string &name) const {
  auto it = attrs_.find(name);
  PADDLE_ENFORCE(it != attrs_.end(), "Attribute %s is not found", name);
  return static_cast<AttrType>(it->second.which() - 1);
}

std::vector<std::string> OpDescBind::AttrNames() const {
  std::vector<std::string> retv;
  retv.reserve(attrs_.size());
  for (auto &attr : attrs_) {
    retv.push_back(attr.first);
  }
  return retv;
}

void OpDescBind::SetAttr(const std::string &name, const Attribute &v) {
  this->attrs_[name] = v;
  need_update_ = true;
}

void OpDescBind::SetBlockAttr(const std::string &name, BlockDescBind &block) {
  BlockDesc *desc = block.Proto();
  this->attrs_[name] = desc;
  need_update_ = true;
}

void OpDescBind::SetAttrMap(
    const std::unordered_map<std::string, Attribute> &attr_map) {
  attrs_ = attr_map;
  need_update_ = true;
}

Attribute OpDescBind::GetAttr(const std::string &name) const {
  auto it = attrs_.find(name);
  PADDLE_ENFORCE(it != attrs_.end(), "Attribute %s is not found", name);
  return it->second;
}

int OpDescBind::GetBlockAttr(const std::string &name) const {
  auto it = attrs_.find(name);
  PADDLE_ENFORCE(it != attrs_.end(), "Attribute %s is not found", name);
  return boost::get<BlockDesc *>(it->second)->idx();
}

const std::unordered_map<std::string, Attribute> &OpDescBind::GetAttrMap()
    const {
  return attrs_;
}

void OpDescBind::Rename(const std::string &old_name,
                        const std::string &new_name) {
  for (auto &input : inputs_) {
    std::replace(input.second.begin(), input.second.end(), old_name, new_name);
  }
  for (auto &output : outputs_) {
    std::replace(output.second.begin(), output.second.end(), old_name,
                 new_name);
  }
  need_update_ = true;
}

struct SetAttrDescVisitor : public boost::static_visitor<void> {
  explicit SetAttrDescVisitor(OpDesc::Attr *attr) : attr_(attr) {}
  mutable OpDesc::Attr *attr_;
  void operator()(int v) const { attr_->set_i(v); }
  void operator()(float v) const { attr_->set_f(v); }
  void operator()(const std::string &v) const { attr_->set_s(v); }
  void operator()(bool b) const { attr_->set_b(b); }

  void operator()(const std::vector<int> &v) const {
    VectorToRepeated(v, attr_->mutable_ints());
  }
  void operator()(const std::vector<float> &v) const {
    VectorToRepeated(v, attr_->mutable_floats());
  }
  void operator()(const std::vector<std::string> &v) const {
    VectorToRepeated(v, attr_->mutable_strings());
  }
  void operator()(const std::vector<bool> &v) const {
    VectorToRepeated(v, attr_->mutable_bools());
  }
  void operator()(BlockDesc *desc) const { attr_->set_block_idx(desc->idx()); }
  void operator()(boost::blank) const { PADDLE_THROW("Unexpected branch"); }
};

void OpDescBind::Flush() {
  if (need_update_) {
    this->desc_.mutable_inputs()->Clear();
    for (auto &ipt : inputs_) {
      auto *input = desc_.add_inputs();
      input->set_parameter(ipt.first);
      VectorToRepeated(ipt.second, input->mutable_arguments());
    }

    this->desc_.mutable_outputs()->Clear();
    for (auto &opt : outputs_) {
      auto *output = desc_.add_outputs();
      output->set_parameter(opt.first);
      VectorToRepeated(opt.second, output->mutable_arguments());
    }

    this->desc_.mutable_attrs()->Clear();
    for (auto &attr : attrs_) {
      auto *attr_desc = desc_.add_attrs();
      attr_desc->set_name(attr.first);
      attr_desc->set_type(
          static_cast<framework::AttrType>(attr.second.which() - 1));
      SetAttrDescVisitor visitor(attr_desc);
      boost::apply_visitor(visitor, attr.second);
    }

    need_update_ = false;
  }
}

using InferShapeFuncMap =
    std::unordered_map<std::string /*op_type*/,
                       std::function<void(InferShapeContext *)>>;

static InferShapeFuncMap &InferShapeFuncs() {
  static InferShapeFuncMap *g_map = nullptr;
  if (g_map == nullptr) {
    g_map = new InferShapeFuncMap();
    auto &info_map = OpInfoMap::Instance();
    // all registered kernels
    for (auto &pair : OperatorWithKernel::AllOpKernels()) {
      auto &info = info_map.Get(pair.first);
      // use empty type here to avoid runtime checks.
      auto op =
          static_cast<OperatorWithKernel *>(info.Creator()("", {}, {}, {}));
      g_map->insert(
          {pair.first, [op](InferShapeContext *ctx) { op->InferShape(ctx); }});
    }
  }
  return *g_map;
}

void OpDescBind::CheckAttrs() {
  PADDLE_ENFORCE(!Type().empty(),
                 "CheckAttr() can not be called before type is setted.");
  auto *checker = OpInfoMap::Instance().Get(Type()).Checker();
  if (checker == nullptr) {
    // checker is not configured. That operator could be generated by Paddle,
    // not by users.
    return;
  }
  checker->Check(attrs_);
}

void OpDescBind::InferShape(const BlockDescBind &block) const {
  auto &funcs = InferShapeFuncs();
  auto it = funcs.find(this->Type());
  if (it == funcs.end()) {
    PADDLE_THROW("Operator %s has not been registered", this->Type());
  }
  CompileTimeInferShapeContext ctx(*this, block);
  it->second(&ctx);
}

void OpDescBind::InferVarType(BlockDescBind *block) const {
  auto &info = OpInfoMap::Instance().Get(this->Type());
  if (info.infer_var_type_) {
    info.infer_var_type_(*this, block);
  } else {
    // all output type is LoDTensor by default
    for (auto &out_pair : this->outputs_) {
      for (auto &out_var_name : out_pair.second) {
        block->Var(out_var_name)->SetType(VarDesc::LOD_TENSOR);
      }
    }
  }
}

}  // namespace framework
}  // namespace paddle
