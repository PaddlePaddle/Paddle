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

 private:
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
    void operator()(BlockDesc *desc) const {
      attr_->set_block_idx(desc->idx());
    }
    void operator()(boost::blank) const { PADDLE_THROW("Unexpected branch"); }
  };

  void Sync();

  OpDesc op_desc_;
  std::unordered_map<std::string, std::vector<std::string>> inputs_;
  std::unordered_map<std::string, std::vector<std::string>> outputs_;
  std::unordered_map<std::string, Attribute> attrs_;

  // need_update_ indicate there some local changes not be synchronized. If
  // local changes should be synchronized, need_update_ should be set to true.
  bool need_update_{false};
};
}  // namespace framework
}  // namespace paddle
