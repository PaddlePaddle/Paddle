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

#include <deque>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include "paddle/framework/op_desc.h"
#include "paddle/framework/proto_desc.h"
#include "paddle/framework/var_desc.h"
#include "paddle/platform/macros.h"

namespace paddle {
namespace framework {

class ProgramDescBind;

// Each Protobuf Message, we provide a XXXBind class. In that class, we optimize
// read/write speed. Only when we want the protobuf message, the local changes
// will be synchronized (by `Sync` method).

class BlockDescBind {
 public:
  BlockDescBind(ProgramDescBind *prog, BlockDesc *desc);

  BlockDescBind(const BlockDescBind &other, BlockDesc *desc,
                ProgramDescBind *prog);

  ~BlockDescBind() {
    this->ClearPBVars();
    this->ClearPBOps();
  }

  int32_t ID() const { return desc_->idx(); }

  int32_t Parent() const { return desc_->parent_idx(); }

  VarDescBind *Var(const std::string &name_bytes);

  VarDescBind *FindVar(const std::string &name_bytes) const;

  bool HasVar(const std::string &var_name) const;

  VarDescBind *FindVarRecursive(const std::string &name_bytes) const;

  VarDescBind *FindRecursiveOrCreateVar(const std::string &name_bytes);

  bool HasVarRecursive(const std::string &var_name) const;

  std::set<std::string> LocalVarNames() const {
    std::set<std::string> var_names;
    for (auto &var : vars_) {
      var_names.insert(var.first);
    }
    return var_names;
  }

  std::vector<VarDescBind *> AllVars() const;

  BlockDescBind *ParentBlock() const;

  OpDescBind *AppendOp();

  void AppendAllocatedOp(std::unique_ptr<OpDescBind> &&op_desc);

  OpDescBind *PrependOp();

  std::vector<OpDescBind *> AllOps() const;

  size_t OpSize() const { return ops_.size(); }

  OpDescBind *Op(int idx) { return ops_.at(idx).get(); }

  void Flush();

  BlockDesc *Proto();

  ProgramDescBind *Program() { return this->prog_; }

 private:
  void ClearPBOps();
  void ClearPBVars();

 private:
  ProgramDescBind *prog_;  // not_own
  BlockDesc *desc_;        // not_own
  bool need_update_;

  std::deque<std::unique_ptr<OpDescBind>> ops_;
  std::unordered_map<std::string, std::unique_ptr<VarDescBind>> vars_;

  DISABLE_COPY_AND_ASSIGN(BlockDescBind);
};
}  // namespace framework
}  // namespace paddle
