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

#include "paddle/framework/block.h"
#include "paddle/platform/enforce.h"

namespace paddle {
namespace framework {

const OpDesc* SymbolTable::NewOp() {
  ops_.emplace_back();
  return &ops_.back();
}

const VarDesc* SymbolTable::NewVar(const std::string& name) {
  PADDLE_ENFORCE(vars_.insert(std::make_pair(name, VarDesc())).second,
                 "Var called %s duplicates", name.c_str());
  auto& var = vars_[name];
  var.set_name(name);
  return &var;
}

const VarDesc* SymbolTable::FindVar(const std::string& name,
                                    bool recursive) const {
  auto it = vars_.find(name);
  if (it != vars_.end()) {
    return &(it->second);
  }
  if (recursive && parent_) {
    return parent_->FindVar(name, true);
  }
  // not found
  return nullptr;
}

const OpDesc* SymbolTable::FindOp(size_t idx) const {
  PADDLE_ENFORCE_LT(idx, ops_.size());
  return &ops_[idx];
}

BlockDesc SymbolTable::Compile() const {
  BlockDesc res;
  for (const auto& op : ops_) {
    *res.add_ops() = op;
  }
  for (const auto& it : vars_) {
    *res.add_vars() = it.second;
  }
  return res;
}

}  // namespace framework
}  // namespace paddle
