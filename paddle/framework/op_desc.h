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

#pragma once

#include <unordered_map>
#include <vector>
#include "paddle/framework/attribute.h"
#include "paddle/framework/type_defs.h"
#include "paddle/framework/var_desc.h"

namespace paddle {
namespace framework {

class BlockDescBind;
class ProgramDescBind;

class OpDescBind {
 public:
  OpDescBind() {}

  OpDescBind(const std::string &type, const VariableNameMap &inputs,
             const VariableNameMap &outputs, const AttributeMap &attrs);

  OpDescBind(const OpDesc &desc, ProgramDescBind *prog);

  OpDesc *Proto();

  std::string Type() const { return desc_.type(); }

  void SetType(const std::string &type) { desc_.set_type(type); }

  const std::vector<std::string> &Input(const std::string &name) const;

  std::vector<std::string> InputArgumentNames() const;

  void SetInput(const std::string &param_name,
                const std::vector<std::string> &args);

  const std::vector<std::string> &Output(const std::string &name) const;

  std::vector<std::string> OutputArgumentNames() const;

  void SetOutput(const std::string &param_name,
                 const std::vector<std::string> &args);

  bool HasAttr(const std::string &name) const {
    return attrs_.find(name) != attrs_.end();
  }

  AttrType GetAttrType(const std::string &name) const;

  std::vector<std::string> AttrNames() const;

  void SetAttr(const std::string &name, const Attribute &v);

  void SetBlockAttr(const std::string &name, BlockDescBind &block);

  Attribute GetAttr(const std::string &name) const;

  int GetBlockAttr(const std::string &name) const;

  void Rename(const std::string &old_name, const std::string &new_name);

  // Only be used in C++
  const AttributeMap &GetAttrMap() const;

  // Only be used in C++
  void SetAttrMap(const AttributeMap &attr_map);

  std::vector<std::string> InputNames() const { return MapKeys(inputs_); }
  std::vector<std::string> OutputNames() const { return MapKeys(outputs_); }

  void SetInputMap(const VariableNameMap &input) {
    this->inputs_ = input;
    this->need_update_ = true;
  }

  void SetOutputMap(const VariableNameMap &output) {
    this->outputs_ = output;
    this->need_update_ = true;
  }

  const VariableNameMap &Inputs() const { return inputs_; }

  const VariableNameMap &Outputs() const { return outputs_; }

  AttributeMap *MutableAttrMap() {
    this->need_update_ = true;
    return &this->attrs_;
  }

  void CheckAttrs();

  void InferShape(const BlockDescBind &block) const;

  void InferVarType(BlockDescBind *block) const;

  void MarkAsTarget() { desc_.set_is_target(true); }

  void Flush();

 private:
  template <typename MapType>
  static std::vector<typename MapType::key_type> MapKeys(const MapType &map) {
    std::vector<typename MapType::key_type> ret_val;
    ret_val.reserve(map.size());
    std::transform(
        map.begin(), map.end(), std::back_inserter(ret_val),
        [](const typename MapType::value_type &pair) { return pair.first; });
    return ret_val;
  }

  OpDesc desc_;
  VariableNameMap inputs_;
  VariableNameMap outputs_;
  AttributeMap attrs_;

  // need_update_ indicate there some local changes not be synchronized. If
  // local changes should be synchronized, need_update_ should be set to true.
  bool need_update_{false};
};
}  // namespace framework
}  // namespace paddle
