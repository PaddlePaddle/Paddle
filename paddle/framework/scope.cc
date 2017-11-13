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

#include "paddle/framework/scope.h"

#include <memory>  // for unique_ptr
#include <mutex>   // for call_once
#include "glog/logging.h"
#include "paddle/string/printf.h"

namespace paddle {
namespace framework {

Scope::~Scope() {
  DropKids();
  for (auto& kv : vars_) {
    VLOG(3) << "Destroy variable " << kv.first;
    delete kv.second;
  }
}

Scope& Scope::NewScope() const {
  kids_.push_back(new Scope(this));
  return *kids_.back();
}

Variable* Scope::Var(const std::string& name) {
  auto iter = vars_.find(name);
  if (iter != vars_.end()) {
    return iter->second;
  }
  Variable* v = new Variable();
  vars_[name] = v;
  VLOG(3) << "Create variable " << name << " on scope";
  v->name_ = &(vars_.find(name)->first);
  return v;
}

Variable* Scope::Var(std::string* name) {
  auto var_name = string::Sprintf("%p.%d", this, vars_.size());
  if (name != nullptr) {
    *name = var_name;
  }
  return Var(var_name);
}

Variable* Scope::FindVar(const std::string& name) const {
  auto it = vars_.find(name);
  if (it != vars_.end()) return it->second;
  return (parent_ == nullptr) ? nullptr : parent_->FindVar(name);
}

const Scope* Scope::FindScope(const Variable* var) const {
  for (auto& kv : vars_) {
    if (kv.second == var) {
      return this;
    }
  }
  return (parent_ == nullptr) ? nullptr : parent_->FindScope(var);
}
void Scope::DropKids() {
  for (Scope* s : kids_) delete s;
  kids_.clear();
}

std::vector<std::string> Scope::GetAllNames(bool recursive) const {
  std::vector<std::string> known_vars(vars_.size());

  if (recursive) {
    for (auto& kid : kids_) {
      auto kid_vars = kid->GetAllNames();
      for (auto& p : kid_vars) {
        known_vars.emplace_back(p);
      }
    }
  }
  for (auto& p : vars_) {
    known_vars.emplace_back(p.first);
  }
  return known_vars;
}

void Scope::DeleteScope(Scope* scope) {
  auto it = std::find(this->kids_.begin(), this->kids_.end(), scope);
  PADDLE_ENFORCE(it != this->kids_.end(), "Cannot find %p as kid scope", scope);
  this->kids_.erase(it);
  delete scope;
}

void Scope::Rename(const std::string& origin_name,
                   const std::string& new_name) const {
  auto origin_it = vars_.find(origin_name);
  PADDLE_ENFORCE(origin_it != vars_.end(),
                 "Cannot find original variable with name %s", origin_name);
  auto new_it = vars_.find(new_name);
  PADDLE_ENFORCE(new_it == vars_.end(),
                 "The variable with name %s is already in the scope", new_name);
  vars_[new_name] = origin_it->second;
  vars_.erase(origin_it);
}

std::string Scope::Rename(const std::string& origin_name) const {
  auto var_name = string::Sprintf("%p.%d", this, vars_.size());
  Rename(origin_name, var_name);
  return var_name;
}

}  // namespace framework
}  // namespace paddle
