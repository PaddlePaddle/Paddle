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

#include "paddle/framework/block_desc.h"
#include "paddle/framework/program_desc.h"

namespace paddle {
namespace framework {

VarDescBind *BlockDescBind::Var(const std::string &name) {
  need_update_ = true;
  auto it = vars_.find(name);
  if (it != vars_.end()) {
    return it->second.get();
  }
  auto *var = new VarDescBind(name);
  vars_[name].reset(var);
  return var;
}

VarDescBind *BlockDescBind::FindVar(const std::string &name) const {
  auto it = vars_.find(name);
  if (it == vars_.end()) {
    return nullptr;
  }
  return it->second.get();
}

bool BlockDescBind::HasVar(const std::string &name) const {
  return vars_.find(name) != vars_.end();
}

std::vector<VarDescBind *> BlockDescBind::AllVars() const {
  std::vector<VarDescBind *> res;
  for (const auto &p : vars_) {
    res.push_back(p.second.get());
  }
  return res;
}

OpDescBind *BlockDescBind::AppendOp() {
  need_update_ = true;
  ops_.emplace_back(new OpDescBind());
  return ops_.back().get();
}

OpDescBind *BlockDescBind::PrependOp() {
  need_update_ = true;
  ops_.emplace_front(new OpDescBind());
  return ops_.front().get();
}

std::vector<OpDescBind *> BlockDescBind::AllOps() const {
  std::vector<OpDescBind *> res;
  for (const auto &op : ops_) {
    res.push_back(op.get());
  }
  return res;
}

void BlockDescBind::Flush() {
  if (need_update_) {
    auto &op_field = *this->desc_->mutable_ops();
    op_field.Clear();
    op_field.Reserve(static_cast<int>(ops_.size()));
    for (auto &op_desc : ops_) {
      op_field.AddAllocated(op_desc->Proto());
    }
    auto &var_field = *this->desc_->mutable_vars();
    var_field.Clear();
    var_field.Reserve(static_cast<int>(vars_.size()));
    for (auto &var_desc : vars_) {
      var_field.AddAllocated(var_desc.second->Proto());
    }
    need_update_ = false;
  }
}

BlockDescBind *BlockDescBind::ParentBlock() const {
  if (this->desc_->parent_idx() == -1) {
    return nullptr;
  }
  return prog_->Block(static_cast<size_t>(this->desc_->parent_idx()));
}

BlockDesc *BlockDescBind::Proto() {
  Flush();
  return desc_;
}

}  // namespace framework
}  // namespace paddle
