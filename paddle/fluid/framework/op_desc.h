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
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/common/macros.h"
#include "paddle/fluid/distributed/auto_parallel/dist_attr.h"
#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/utils/test_macros.h"

namespace paddle {
namespace framework {

class VarDesc;
class BlockDesc;
class ProgramDesc;

using paddle::distributed::auto_parallel::OperatorDistAttr;

class TEST_API OpDesc {
 public:
  OpDesc();

  ~OpDesc();

  OpDesc(const std::string &type,
         const VariableNameMap &inputs,
         const VariableNameMap &outputs,
         const AttributeMap &attrs);

  OpDesc(const OpDesc &desc);

  OpDesc(const proto::OpDesc &desc, BlockDesc *block);

  explicit OpDesc(BlockDesc *block) : block_(block) {}

  OpDesc(const OpDesc &other, BlockDesc *block);

  OpDesc &operator=(const OpDesc &other);

  void CopyFrom(const OpDesc &op_desc);

  proto::OpDesc *Proto();

  std::string Type() const { return desc_.type(); }

  void SetType(const std::string &type);

  const std::vector<std::string> &Input(const std::string &name) const;

  std::vector<std::string> Input(const std::string &name,
                                 bool with_attr_var) const;

  std::vector<std::string> InputArgumentNames(bool with_attr_var = false) const;

  void SetInput(const std::string &param_name,
                const std::vector<std::string> &args);

  const std::vector<std::string> &Output(const std::string &name) const;

  bool HasOutput(const std::string &name) const;

  bool HasInput(const std::string &name, bool with_attr_var = false) const;

  std::vector<std::string> OutputArgumentNames() const;

  void SetOutput(const std::string &param_name,
                 const std::vector<std::string> &args);
  void RemoveOutput(const std::string &name);

  void RemoveInput(const std::string &name);

  bool HasAttr(const std::string &name, bool with_attr_var = false) const;

  bool HasProtoAttr(const std::string &name) const;

  proto::AttrType GetAttrType(const std::string &name,
                              bool with_attr_var = false) const;

  std::vector<std::string> AttrNames(bool with_attr_var = false) const;

  void SetAttr(const std::string &name, const Attribute &v);
  void RemoveAttr(const std::string &name);

  // NOTE(chenfeiyu): this template is added to avoid using a variant(Attribute)
  // as a parameter of a function which is bound to python, which causes
  // unexpected type conversion due to the overload resolution mechanism
  // https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html#c-17-library-containers
  template <typename T>
  void SetPlainAttr(const std::string &name, const T &value) {
    SetAttr(name, value);
  }

  void SetVarAttr(const std::string &name, VarDesc *var);

  void SetVarsAttr(const std::string &name, std::vector<VarDesc *> vars);

  void SetBlockAttr(const std::string &name, BlockDesc *block);

  void SetBlocksAttr(const std::string &name, std::vector<BlockDesc *> blocks);

  Attribute GetAttr(const std::string &name, bool with_attr_var = false) const;

  template <typename T>
  T GetAttrIfExists(const std::string &name) const {
    T result{};
    if (HasAttr(name)) {
      result = PADDLE_GET_CONST(T, GetAttr(name));
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

  void SetRuntimeAttrMap(const AttributeMap &attr_map);

  const AttributeMap &GetRuntimeAttrMap() const;

  std::vector<std::string> InputNames(bool with_attr_var UNUSED = false) const {
    return MapKeys(inputs_);
  }
  std::vector<std::string> OutputNames() const { return MapKeys(outputs_); }

  const VariableNameMap &Inputs() const { return inputs_; }

  VariableNameMap Inputs(bool with_attr_var) const;

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

  void InferShape(const BlockDesc &block);

  void InferVarType(BlockDesc *block) const;

  void SetIsTarget(bool is_target) { desc_.set_is_target(is_target); }

  void Flush();

  BlockDesc *Block() { return this->block_; }

  const BlockDesc *Block() const { return this->block_; }

  void UpdateVarAttr(const std::string &name, const Attribute &attr);

  bool NeedUpdate() const { return need_update_; }

  // The following methods are only used for auto parallel.
  uint64_t Id() const { return id_; }
  uint64_t OriginalId() const { return original_id_; }
  void SetOriginalId(uint64_t original_id) { original_id_ = original_id; }
  const OperatorDistAttr *DistAttr() const;
  OperatorDistAttr *MutableDistAttr();
  void SetDistAttr(const OperatorDistAttr &dist_attr);

  void ResetBlock() { this->block_ = nullptr; }

 private:
  friend class ProgramDesc;
  // Find VarDesc from OpDesc located Block into global Block
  VarDesc *FindVarRecursive(const std::string &name);

  template <typename MapType>
  static std::vector<typename MapType::key_type> MapKeys(const MapType &map) {
    std::vector<typename MapType::key_type> ret_val;
    ret_val.reserve(map.size());
    std::transform(
        map.begin(),
        map.end(),
        std::back_inserter(ret_val),
        [](const typename MapType::value_type &pair) { return pair.first; });
    return ret_val;
  }

  // Is it really needed? Or just maintain a ptr from the block?
  proto::OpDesc desc_;
  BlockDesc *block_{nullptr};  // not_own
  // input arg name => input variable names
  VariableNameMap inputs_;
  // output arg name => output variable names
  VariableNameMap outputs_;
  // attribute name => all original attrs
  AttributeMap attrs_;
  // runtime_attrs_ contains the attributes which used for dispatching kernel
  // (use_mkldnn, use_cudnn, ...) or passing additional configuration for
  // special heterogeneous kernel (workspace_size_MB, ...).
  // The attributes in runtime_attrs_ are setted by framework (such as PASS),
  // and not in the python api.
  AttributeMap runtime_attrs_;

  // need_update_ indicate there some local changes not be synchronized. If
  // local changes should be synchronized, need_update_ should be set to true.
  bool need_update_{false};

  // Note: the following members are only used for auto_parallel for now.
  static uint64_t GenerateId() {
    static std::atomic<std::uint64_t> uid{0};
    // Must start from one
    return ++uid;
  }
  uint64_t id_ = GenerateId();
  uint64_t original_id_ = id_;
  std::unique_ptr<OperatorDistAttr> dist_attr_;
};

std::vector<std::string> AttrVarNames(const Attribute &attr);
}  // namespace framework
}  // namespace paddle
