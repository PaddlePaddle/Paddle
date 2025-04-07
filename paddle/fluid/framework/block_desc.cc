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

#include "paddle/fluid/framework/block_desc.h"

#include <queue>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle::framework {

VarDesc *BlockDesc::Var(const std::string &name) {
  auto it = vars_.find(name);
  if (it != vars_.end()) {
    return it->second.get();
  }
  need_update_ = true;
  auto *var = new VarDesc(name);
  vars_[name].reset(var);
  return var;
}

VarDesc *BlockDesc::FindVar(const std::string &name) const {
  auto it = vars_.find(name);
  if (it == vars_.end()) {
    return nullptr;
  }
  return it->second.get();
}

bool BlockDesc::HasVar(const std::string &name) const {
  return vars_.find(name) != vars_.end();
}

VarDesc *BlockDesc::RenameVar(const std::string &old_name,
                              const std::string &new_name) {
  if (!this->HasVar(old_name)) {
    return nullptr;
  }
  need_update_ = true;
  auto *var = this->Var(old_name);
  VarDesc *new_var = new VarDesc(*(var->Proto()));
  new_var->SetName(new_name);
  vars_[new_name].reset(new_var);
  // rename inputs and outputs
  for (const auto &op : ops_) {
    auto *it = op.get();
    it->Rename(old_name, new_name);
  }
  vars_.erase(old_name);
  return new_var;
}

VarDesc *BlockDesc::FindVarRecursive(const std::string &name) const {
  if (name == kEmptyVarName) return nullptr;

  std::queue<const BlockDesc *> frontier;
  std::unordered_set<const BlockDesc *> visited;

  frontier.push(this);

  while (!frontier.empty()) {  // BFS
    auto cur = frontier.front();
    frontier.pop();
    if (visited.count(cur) != 0) {
      continue;
    }
    auto var = cur->FindVar(name);
    if (var != nullptr) {
      return var;
    }

    auto fwd = cur->ForwardBlock();
    auto parent = cur->ParentBlock();

    if (fwd != nullptr) {
      frontier.push(fwd);
    }
    if (parent != nullptr) {
      frontier.push(parent);
    }

    visited.insert(cur);
  }

  return nullptr;
}

VarDesc &BlockDesc::FindRecursiveOrCreateVar(const std::string &name_bytes) {
  VarDesc *res = FindVarRecursive(name_bytes);
  if (res == nullptr) {
    res = Var(name_bytes);
  }
  return *res;
}

bool BlockDesc::HasVarRecursive(const std::string &name) const {
  return FindVarRecursive(name) != nullptr;
}

std::vector<VarDesc *> BlockDesc::AllVars() const {
  std::vector<VarDesc *> res;
  for (const auto &p : vars_) {
    res.push_back(p.second.get());
  }
  return res;
}

OpDesc *BlockDesc::AppendOp() {
  need_update_ = true;
  ops_.emplace_back(new OpDesc(this));
  return ops_.back().get();
}

void BlockDesc::AppendAllocatedOp(std::unique_ptr<OpDesc> &&op_desc) {
  need_update_ = true;
  ops_.emplace_back(std::move(op_desc));
}

OpDesc *BlockDesc::PrependOp() {
  need_update_ = true;
  ops_.emplace_front(new OpDesc(this));
  return ops_.front().get();
}

void BlockDesc::PrependAllocatedOp(std::unique_ptr<OpDesc> &&op_desc) {
  need_update_ = true;
  ops_.emplace_front(std::move(op_desc));
}

OpDesc *BlockDesc::InsertOp(size_t index) {
  need_update_ = true;
  auto it = ops_.begin() + index;  // NOLINT
  std::unique_ptr<OpDesc> new_op(new OpDesc(this));
  it = ops_.insert(it, std::move(new_op));
  return (*it).get();
}

void BlockDesc::RemoveOp(size_t s, size_t e) {
  if (ops_.begin() + s >= ops_.end() ||  // NOLINT
      ops_.begin() + e > ops_.end()) {   // NOLINT
    return;
  }
  need_update_ = true;
  ops_.erase(ops_.begin() + s, ops_.begin() + e);  // NOLINT
}

void BlockDesc::RemoveOpInternal(const OpDesc *op_desc) {
  // TODO(minqiyang): make this faster
  for (auto it = ops_.begin(); it != ops_.end(); ++it) {
    if (it->get() == op_desc) {
      ops_.erase(it);
      need_update_ = true;
      break;
    }
  }
}
void BlockDesc::RemoveVar(const std::string &name) {
  need_update_ = true;
  vars_.erase(name);
}
std::vector<OpDesc *> BlockDesc::AllOps() const {
  std::vector<OpDesc *> res;
  for (const auto &op : ops_) {
    res.push_back(op.get());
  }
  return res;
}

void BlockDesc::Flush() {
  auto need_update = NeedUpdate(true);
  for (auto &op_desc : ops_) {
    op_desc->Flush();
  }
  // no flush for var_desc? or is op_desc flush really needed?
  VLOG(10) << "Flush " << NeedUpdate(true) << " " << need_update << std::endl;
  if (need_update) {
    this->desc_->mutable_ops()->Clear();
    for (auto &op_desc : ops_) {
      this->desc_->mutable_ops()->Add()->CopyFrom(*op_desc->Proto());
      // op_desc's need_update is set to false in op_desc->Flush();
    }

    std::vector<std::string> var_names;
    std::set<std::string> var_names_set;

    // keep order
    for (const auto &var : this->desc_->vars()) {
      var_names.emplace_back(var.name());
      var_names_set.insert(var.name());
    }
    VLOG(4) << "vars in desc " << this->desc_->vars().size();
    this->desc_->mutable_vars()->Clear();
    for (const auto &name : var_names) {
      if (vars_.count(name)) {
        VLOG(4) << "Flush " << name;
        this->desc_->mutable_vars()->Add()->CopyFrom(*vars_[name]->Proto());
        vars_[name]->SetNeedUpdate(false);
      }
    }

    for (auto &var_desc : vars_) {
      if (var_names_set.count(var_desc.first) != 1) {
        VLOG(4) << "Flush " << var_desc.first;
        this->desc_->mutable_vars()->Add()->CopyFrom(*var_desc.second->Proto());
        var_desc.second->SetNeedUpdate(false);
      }
    }

    // this->desc_->mutable_vars()->Clear();
    // for (auto &var_desc : vars_) {
    //   this->desc_->mutable_vars()->Add()->CopyFrom(*var_desc.second->Proto());
    //   var_desc.second->SetNeedUpdate(false);
    // }
    need_update_ = false;
  }
}

BlockDesc *BlockDesc::ParentBlock() const {
  return prog_->MutableBlock(static_cast<size_t>(desc_->parent_idx()));
}

proto::BlockDesc *BlockDesc::Proto() {
  Flush();
  return desc_;
}

BlockDesc::BlockDesc(ProgramDesc *prog, proto::BlockDesc *desc)
    : prog_(prog), desc_(desc), need_update_(false) {
  for (const proto::VarDesc &var_desc : desc_->vars()) {
    vars_[var_desc.name()] = std::make_unique<VarDesc>(var_desc);
  }

  for (const proto::OpDesc &op_desc : desc_->ops()) {
    ops_.emplace_back(new OpDesc(op_desc, this));
  }
}

BlockDesc::BlockDesc(const BlockDesc &other,
                     proto::BlockDesc *desc,
                     ProgramDesc *prog)
    : prog_(prog), desc_(desc) {
  need_update_ = true;
  // NOTE(dev): Init vars_ firstly so we can find them
  // while constructing OpDesc.
  for (auto &it : other.vars_) {
    auto *var = new VarDesc(*it.second);
    vars_[it.first].reset(var);
  }
  for (auto &op : other.ops_) {
    ops_.emplace_back(new OpDesc(*op, this));
  }
}

void BlockDesc::SetForwardBlockID(int32_t forward_block_id) {
  PADDLE_ENFORCE_EQ(
      desc_->has_forward_block_idx(),
      false,
      common::errors::PreconditionNotMet(
          "Block %d's parent block ID has been set to %d, cannot be set to %d.",
          desc_->idx(),
          desc_->forward_block_idx(),
          forward_block_id));
  desc_->set_forward_block_idx(forward_block_id);
}

BlockDesc *BlockDesc::ForwardBlock() const {
  return prog_->MutableBlock(static_cast<size_t>(desc_->forward_block_idx()));
}

void BlockDesc::MoveFrom(BlockDesc *block) {
  PADDLE_ENFORCE_NOT_NULL(
      block, common::errors::InvalidArgument("Block must be provided."));
  if (this == block) {
    return;
  }

  for (auto &pair : block->vars_) {
    const auto &name = pair.first;
    auto &var_ptr = pair.second;
    auto &old_var_ptr = vars_[name];
    if (old_var_ptr == nullptr) {
      VLOG(10) << "Create new variable " << var_ptr->Name();
      old_var_ptr = std::move(var_ptr);
    } else {
      // NOTE(zjl): cannot release old_var_ptr, because Python
      // Variable holds the reference of the C++ VarDesc object.
      // If the C++ VarDesc object is destructed, any call to the
      // methods of Python Variable may raise segmentation fault.
      VLOG(10) << "Update old variable " << var_ptr->Name();
      *old_var_ptr = *var_ptr;
    }
  }
  ops_.clear();
  for (const auto &src_op : block->ops_) {
    auto *dst_op = AppendOp();
    dst_op->CopyFrom(*src_op);
    for (const auto &pair : src_op->GetAttrMap()) {
      const auto &attr_name = pair.first;
      const auto &attr_value = pair.second;
      auto attr_type = static_cast<proto::AttrType>(attr_value.index() - 1);
      if (attr_type == proto::AttrType::VAR ||
          attr_type == proto::AttrType::VARS) {
        dst_op->UpdateVarAttr(attr_name, attr_value);
      } else if (attr_type == proto::AttrType::BLOCK) {
        ProgramDesc *program = block->Program();
        std::vector<framework::BlockDesc *> old_block_desc;
        // NOTE(GhostScreaming): don't use program->proto()->blocks_size(),
        // previous assignment of new Variable in vars_ use std::move,
        // which makes 'var_ptr' which held by 'block' a nullptr.
        // block->Program()->proto() will calls Flush() at first,
        // a null var_ptr will cause segmentation fault.
        int block_size = static_cast<int>(program->Size());
        for (int i = 0; i < block_size; ++i) {
          // record all block desc's ptr from origin block's program
          old_block_desc.emplace_back(program->MutableBlock(i));
        }
        framework::BlockDesc *block_desc =
            PADDLE_GET_CONST(BlockDesc *, attr_value);
        if (std::find(old_block_desc.begin(),
                      old_block_desc.end(),
                      block_desc) != old_block_desc.end()) {
          // The block is owned by the origin block's program. Just use id to
          // get the corresponding block.
          auto block_id = block_desc->ID();
          dst_op->SetBlockAttr(attr_name, prog_->MutableBlock(block_id));
          VLOG(10) << "Set block attr " << attr_name << " id " << block_id;
        } else {
          // The block is not owned by the origin block's program. Should copy
          // the real block desc instead of logical block in the program.
          dst_op->SetBlockAttr(attr_name, block_desc);
          VLOG(10) << "Set block attr " << attr_name << " from attr_value";
        }
      } else if (attr_type == proto::AttrType::BLOCKS) {
        auto old_blocks =
            PADDLE_GET_CONST(std::vector<BlockDesc *>, attr_value);
        std::vector<BlockDesc *> new_blocks;
        new_blocks.reserve(old_blocks.size());
        for (auto *b : old_blocks) {
          VLOG(10) << "Set block attr " << attr_name << " id " << b->ID();
          new_blocks.push_back(prog_->MutableBlock(b->ID()));
        }
        dst_op->SetBlocksAttr(attr_name, new_blocks);
      }
    }
  }
  need_update_ = true;
  Flush();

  block->ops_.clear();
  block->vars_.clear();
  block->need_update_ = true;
  block->Flush();
}

bool BlockDesc::NeedUpdate(bool include_subs) {
  bool need = need_update_;
  if (include_subs) {
    for (const auto &op : ops_) {
      if (op->NeedUpdate()) {
        need = true;
        break;
      }
    }
    for (const auto &pair : vars_) {
      if (pair.second->NeedUpdate()) {
        need = true;
        break;
      }
    }
  }
  return need;
}

}  // namespace paddle::framework
