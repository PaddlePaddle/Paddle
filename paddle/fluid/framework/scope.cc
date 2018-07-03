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

#include "paddle/fluid/framework/scope.h"

#include <memory>  // for unique_ptr
#include <set>
#include "glog/logging.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/string/printf.h"

DEFINE_bool(benchmark, false,
            "Doing memory benchmark. It will make deleting scope synchronized, "
            "and add some memory usage logs."
            "Default cuda is asynchronous device, set to True will"
            "force op run in synchronous mode.");

DEFINE_bool(
    eager_delete_scope, true,
    "Delete local scope eagerly. It will reduce GPU memory usage but "
    "slow down the destruction of variables.(around 1% performance harm)");

namespace paddle {
namespace framework {

Scope::~Scope() { DropKids(); }

Scope& Scope::NewScope() const {
  std::unique_lock<std::mutex> lock(mutex_);
  kids_.push_back(new Scope(this));
  return *kids_.back();
}

Variable* Scope::Var(const std::string& name) {
  std::unique_lock<std::mutex> lock(mutex_);
  return VarInternal(name);
}

Variable* Scope::Var(std::string* name) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto new_name = string::Sprintf("%p.%d", this, vars_.size());
  if (name != nullptr) {
    *name = new_name;
  }
  return VarInternal(new_name);
}

Variable* Scope::FindVar(const std::string& name) const {
  std::unique_lock<std::mutex> lock(mutex_);
  return FindVarInternal(name);
}

const Scope* Scope::FindScope(const Variable* var) const {
  std::unique_lock<std::mutex> lock(mutex_);
  return FindScopeInternal(var);
}

void Scope::DropKids() {
  std::unique_lock<std::mutex> lock(mutex_);
  for (Scope* s : kids_) delete s;
  kids_.clear();
}

std::vector<std::string> Scope::LocalVarNames() const {
  std::unique_lock<std::mutex> lock(mutex_);
  std::vector<std::string> known_vars;
  known_vars.reserve(this->vars_.size());
  for (auto& p : vars_) {
    known_vars.emplace_back(p.first);
  }
  return known_vars;
}

void Scope::DeleteScope(Scope* scope) const {
  std::unique_lock<std::mutex> lock(mutex_);
  auto it = std::find(this->kids_.begin(), this->kids_.end(), scope);
  PADDLE_ENFORCE(it != this->kids_.end(), "Cannot find %p as kid scope", scope);
  this->kids_.erase(it);
  // When making memory benchmark on Fluid, we have to delete scope sync.
  if (FLAGS_benchmark || FLAGS_eager_delete_scope) {
    delete scope;
  } else {
    Async([scope] { delete scope; });
  }
}

void Scope::EraseVars(const std::vector<std::string>& var_names) {
  std::unique_lock<std::mutex> lock(mutex_);
  std::set<std::string> var_set(var_names.begin(), var_names.end());
  for (auto it = vars_.begin(); it != vars_.end();) {
    if (var_set.find(it->first) != var_set.end()) {
      it = vars_.erase(it);
    } else {
      ++it;
    }
  }
}

void Scope::Rename(const std::string& origin_name,
                   const std::string& new_name) const {
  std::unique_lock<std::mutex> lock(mutex_);
  RenameInternal(origin_name, new_name);
}

std::string Scope::Rename(const std::string& origin_name) const {
  std::unique_lock<std::mutex> lock(mutex_);
  auto new_name = string::Sprintf("%p.%d", this, vars_.size());
  RenameInternal(origin_name, new_name);
  return new_name;
}

Variable* Scope::VarInternal(const std::string& name) {
  auto* v = FindVarLocally(name);
  if (v != nullptr) return v;

  v = new Variable();
  vars_[name].reset(v);
  VLOG(3) << "Create variable " << name;
  v->name_ = &(vars_.find(name)->first);
  return v;
}

const Scope* Scope::FindScopeInternal(const Variable* var) const {
  for (auto& kv : vars_) {
    if (kv.second.get() == var) {
      return this;
    }
  }
  return (parent_ == nullptr) ? nullptr : parent_->FindScope(var);
}

void Scope::RenameInternal(const std::string& origin_name,
                           const std::string& new_name) const {
  auto origin_it = vars_.find(origin_name);
  PADDLE_ENFORCE(origin_it != vars_.end(),
                 "Cannot find original variable with name %s", origin_name);
  auto new_it = vars_.find(new_name);
  PADDLE_ENFORCE(new_it == vars_.end(),
                 "The variable with name %s is already in the scope", new_name);
  vars_[new_name].reset(origin_it->second.release());
  vars_.erase(origin_it);
}

Variable* Scope::FindVarInternal(const std::string& name) const {
  auto var = FindVarLocally(name);
  if (var != nullptr) {
    return var;
  }
  return (parent_ == nullptr) ? nullptr : parent_->FindVar(name);
}

Variable* Scope::FindVarLocally(const std::string& name) const {
  auto it = vars_.find(name);
  if (it != vars_.end()) return it->second.get();
  return nullptr;
}

}  // namespace framework
}  // namespace paddle
