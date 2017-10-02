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

class OpDescBind {
 public:
  OpDesc *Proto();

  std::string Type() const { return op_desc_.type(); }

  void SetType(const std::string &type) { op_desc_.set_type(type); }

  const std::vector<std::string> &Input(const std::string &name) const;

  std::vector<std::string> InputNames() const;

  void SetInput(const std::string &param_name,
                const std::vector<std::string> &args);

  const std::vector<std::string> &Output(const std::string &name) const;

  std::vector<std::string> OutputNames() const;

  void SetOutput(const std::string &param_name,
                 const std::vector<std::string> &args);

  std::string DebugString() { return this->Proto()->DebugString(); }

  bool HasAttr(const std::string &name) const {
    return attrs_.find(name) != attrs_.end();
  }

  AttrType GetAttrType(const std::string &name) const;

  std::vector<std::string> AttrNames() const;

  void SetAttr(const std::string &name, const Attribute &v);

  void SetBlockAttr(const std::string &name, BlockDescBind &block);

  Attribute GetAttr(const std::string &name) const;

  int GetBlockAttr(const std::string &name) const;

  // Only be used in C++
  const AttributeMap &GetAttrMap() const;

  // Only be used in C++
  void SetAttrMap(const AttributeMap &attr_map);

  std::vector<std::string> InputParamNames() const { return MapKeys(inputs_); }
  std::vector<std::string> OutputParamNames() const {
    return MapKeys(outputs_);
  }

 private:
  template <typename MapType>
  static std::vector<typename MapType::key_type> MapKeys(const MapType &map) {
    std::vector<typename MapType::key_type> ret_val;
    ret_val.reserve(map.size());
    std::transform(
        map.begin(), map.end(), ret_val.begin(),
        [](const typename MapType::value_type &pair) { return pair.first; });
    return ret_val;
  }

  void Sync();

  OpDesc op_desc_;
  VariableNameMap inputs_;
  VariableNameMap outputs_;
  AttributeMap attrs_;

  // need_update_ indicate there some local changes not be synchronized. If
  // local changes should be synchronized, need_update_ should be set to true.
  bool need_update_{false};
};
}  // namespace framework
}  // namespace paddle
