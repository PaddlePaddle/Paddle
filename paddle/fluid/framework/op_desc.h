/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <atomic>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/var_desc.h"

namespace paddle {
namespace framework {

class BlockDesc;
class ProgramDesc;

class OpDesc {
 public:
  OpDesc() {}

  OpDesc(const std::string &type, const VariableNameMap &inputs,
         const VariableNameMap &outputs, const AttributeMap &attrs);

  OpDesc(const proto::OpDesc &desc, BlockDesc *block);

  explicit OpDesc(BlockDesc *block) : block_(block) {}

  OpDesc(const OpDesc &other, BlockDesc *block);

  void CopyFrom(const OpDesc &op_desc);

  proto::OpDesc *Proto();

  std::string Type() const { return desc_.type(); }

  void SetType(const std::string &type) { desc_.set_type(type); }

  const std::vector<std::string> &Input(const std::string &name) const;

  std::vector<std::string> InputArgumentNames() const;

  void SetInput(const std::string &param_name,
                const std::vector<std::string> &args);

  const std::vector<std::string> &Output(const std::string &name) const;

  bool HasOutput(const std::string &name) const;

  std::vector<std::string> OutputArgumentNames() const;

  void SetOutput(const std::string &param_name,
                 const std::vector<std::string> &args);
  void RemoveOutput(const std::string &name);

  void RemoveInput(const std::string &name);

  bool HasAttr(const std::string &name) const {
    return attrs_.find(name) != attrs_.end();
  }

  bool HasProtoAttr(const std::string &name) const;

  proto::AttrType GetAttrType(const std::string &name) const;

  std::vector<std::string> AttrNames() const;

  void SetAttr(const std::string &name, const Attribute &v);
  void RemoveAttr(const std::string &name);

  void SetBlockAttr(const std::string &name, BlockDesc *block);

  void SetBlocksAttr(const std::string &name, std::vector<BlockDesc *> blocks);

  Attribute GetAttr(const std::string &name) const;

  template <typename T>
  T GetAttrIfExists(const std::string &name) const {
    T result{};
    if (HasAttr(name)) {
      result = BOOST_GET_CONST(T, GetAttr(name));
    }
    return result;
  }

  const proto::OpProto::Attr &GetProtoAttr(const std::string &name) const;

  Attribute GetNullableAttr(const std::string &name) const;

  int GetBlockAttrId(const std::string &name) const;

  std::vector<int> GetBlocksAttrIds(const std::string &name) const;

  void Rename(const std::string &old_name, const std::string &new_name);

  void RenameOutput(const std::string &old_name, const std::string &new_name);

  void RenameInput(const std::string &old_name, const std::string &new_name);

  // Only be used in C++
  const AttributeMap &GetAttrMap() const;

  // Only be used in C++
  void SetAttrMap(const AttributeMap &attr_map);

  std::vector<std::string> InputNames() const { return MapKeys(inputs_); }
  std::vector<std::string> OutputNames() const { return MapKeys(outputs_); }

  const VariableNameMap &Inputs() const { return inputs_; }

  const VariableNameMap &Outputs() const { return outputs_; }

  VariableNameMap *MutableInputs() {
    this->need_update_ = true;
    return &this->inputs_;
  }

  VariableNameMap *MutableOutputs() {
    this->need_update_ = true;
    return &this->outputs_;
  }

  AttributeMap *MutableAttrMap() {
    this->need_update_ = true;
    return &this->attrs_;
  }

  void CheckAttrs();

  void InferShape(const BlockDesc &block) const;

  void InferVarType(BlockDesc *block) const;

  void SetIsTarget(bool is_target) { desc_.set_is_target(is_target); }

  void Flush();

  BlockDesc *Block() { return this->block_; }

  const BlockDesc *Block() const { return this->block_; }

  // The Id() and OrignalId() are only used for auto parallel.
  uint64_t Id() const { return id_; }
  uint64_t OriginalId() const { return original_id_; }
  void SetOriginalId(uint64_t original_id) { original_id_ = original_id; }

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

  // This thread-safe implementation seems to be redudent since the neural
  // networks are usually constructed in a single thread
  static uint64_t GenerateId() {
    static std::atomic<std::uint64_t> uid{0};
    // Must start from one
    return ++uid;
  }

  proto::OpDesc desc_;
  BlockDesc *block_{nullptr};  // not_own
  // input arg name => input variable names
  VariableNameMap inputs_;
  // output arg name => output variable names
  VariableNameMap outputs_;
  AttributeMap attrs_;

  // need_update_ indicate there some local changes not be synchronized. If
  // local changes should be synchronized, need_update_ should be set to true.
  bool need_update_{false};

  // Note: the id_ is unique (only for auto parallel).
  uint64_t id_ = GenerateId();
  // Note: the orignal_id_ is used for referring to the original OpDesc
  // that the current OpDesc is built from (only for auto parallel).
  // The default original_id_ is same as the id_, which means the
  // current OpDesc is not built from the other one.
  uint64_t original_id_ = id_;
};
}  // namespace framework
}  // namespace paddle
