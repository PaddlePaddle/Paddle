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
  auto it = vars_.find(name);
  if (it != vars_.end()) {
    return it->second.get();
  }
  need_update_ = true;
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

VarDescBind *BlockDescBind::FindVarRecursive(const std::string &name) const {
  auto it = vars_.find(name);
  if (it == vars_.end()) {
    return Parent() == kNoneBlockIndex ? nullptr
                                       : ParentBlock()->FindVarRecursive(name);
  }
  return it->second.get();
}

VarDescBind *BlockDescBind::FindRecursiveOrCreateVar(
    const std::string &name_bytes) {
  VarDescBind *res = FindVarRecursive(name_bytes);
  if (res == nullptr) {
    res = Var(name_bytes);
  }
  return res;
}

bool BlockDescBind::HasVarRecursive(const std::string &name) const {
  return FindVarRecursive(name) != nullptr;
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

void BlockDescBind::AppendAllocatedOp(std::unique_ptr<OpDescBind> &&op_desc) {
  need_update_ = true;
  ops_.emplace_back(std::move(op_desc));
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
  for (auto &op_desc : ops_) {
    op_desc->Flush();
  }

  if (need_update_) {
    auto &op_field = *this->desc_->mutable_ops();
    this->ClearPBOps();
    op_field.Reserve(static_cast<int>(ops_.size()));
    for (auto &op_desc : ops_) {
      op_field.AddAllocated(op_desc->Proto());
    }
    auto &var_field = *this->desc_->mutable_vars();
    this->ClearPBVars();
    var_field.Reserve(static_cast<int>(vars_.size()));
    for (auto &var_desc : vars_) {
      var_field.AddAllocated(var_desc.second->Proto());
    }
    need_update_ = false;
  }
}

BlockDescBind *BlockDescBind::ParentBlock() const {
  if (this->desc_->parent_idx() == kNoneBlockIndex) {
    return nullptr;
  }
  return prog_->MutableBlock(static_cast<size_t>(this->desc_->parent_idx()));
}

BlockDesc *BlockDescBind::Proto() {
  Flush();
  return desc_;
}

BlockDescBind::BlockDescBind(ProgramDescBind *prog, BlockDesc *desc)
    : prog_(prog), desc_(desc), need_update_(false) {
  for (const VarDesc &var_desc : desc_->vars()) {
    vars_[var_desc.name()].reset(new VarDescBind(var_desc));
  }
  for (const OpDesc &op_desc : desc_->ops()) {
    ops_.emplace_back(new OpDescBind(op_desc, prog));
  }
}

BlockDescBind::BlockDescBind(const BlockDescBind &other, BlockDesc *desc,
                             ProgramDescBind *prog)
    : prog_(prog), desc_(desc) {
  need_update_ = true;
  for (auto &op : other.ops_) {
    ops_.emplace_back(new OpDescBind(*op));
  }

  for (auto &it : other.vars_) {
    auto *var = new VarDescBind(*it.second);
    vars_[it.first].reset(var);
  }
}

void BlockDescBind::ClearPBOps() {
  auto ops = this->desc_->mutable_ops();
  while (!ops->empty()) {
    // we do not own the OpDesc, so release the ownership.
    ops->ReleaseLast();
  }
}

void BlockDescBind::ClearPBVars() {
  auto vars = this->desc_->mutable_vars();
  while (!vars->empty()) {
    // we do not own the VarDesc, so release the ownership.
    vars->ReleaseLast();
  }
}

}  // namespace framework
}  // namespace paddle
