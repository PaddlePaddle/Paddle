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
#include "paddle/framework/block_desc.h"

namespace paddle {
namespace framework {

OpDescBind::OpDescBind(const std::string &type, const VariableNameMap &inputs,
                       const VariableNameMap &outputs,
                       const AttributeMap &attrs) {
  op_desc_.set_type(type);
  inputs_ = inputs;
  outputs_ = outputs;
  attrs_ = attrs;
}

OpDesc *OpDescBind::Proto() {
  Sync();
  return &op_desc_;
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

void OpDescBind::Sync() {
  if (need_update_) {
    this->op_desc_.mutable_inputs()->Clear();
    for (auto &ipt : inputs_) {
      auto *input = op_desc_.add_inputs();
      input->set_parameter(ipt.first);
      VectorToRepeated(ipt.second, input->mutable_arguments());
    }

    this->op_desc_.mutable_outputs()->Clear();
    for (auto &opt : outputs_) {
      auto *output = op_desc_.add_outputs();
      output->set_parameter(opt.first);
      VectorToRepeated(opt.second, output->mutable_arguments());
    }

    this->op_desc_.mutable_attrs()->Clear();
    for (auto &attr : attrs_) {
      auto *attr_desc = op_desc_.add_attrs();
      attr_desc->set_name(attr.first);
      attr_desc->set_type(
          static_cast<framework::AttrType>(attr.second.which() - 1));
      SetAttrDescVisitor visitor(attr_desc);
      boost::apply_visitor(visitor, attr.second);
    }

    need_update_ = false;
  }
}
}  // namespace framework
}  // namespace paddle
