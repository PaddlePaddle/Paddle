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
#include <queue>
#include <set>
#include <unordered_set>
#include "glog/logging.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/string/printf.h"

DECLARE_bool(benchmark);

DEFINE_bool(
    eager_delete_scope, true,
    "Delete local scope eagerly. It will reduce GPU memory usage but "
    "slow down the destruction of variables.(around 1% performance harm)");

// When in inference scenario, the scopes will not be written by two threads in
// a mean time, but a scope may be read by multiple threads concurrently, and
// the mutex will cause serious performance issue.
// So the mutex is disabled when `ON_INFER`.
#ifdef PADDLE_ON_INFERENCE
#define SCOPE_KIDS_READER_LOCK
#define SCOPE_KIDS_WRITER_LOCK
#define SCOPE_VARS_READER_LOCK
#define SCOPE_VARS_WRITER_LOCK
#else
#define SCOPE_KIDS_READER_LOCK AutoRDLock auto_lock(&kids_lock_);
#define SCOPE_KIDS_WRITER_LOCK AutoWRLock auto_lock(&kids_lock_);
#define SCOPE_VARS_READER_LOCK AutoRDLock auto_lock(&vars_lock_);
#define SCOPE_VARS_WRITER_LOCK AutoWRLock auto_lock(&vars_lock_);
#endif

namespace paddle {
namespace framework {

Scope::~Scope() { DropKids(); }

Scope& Scope::NewScope() const {
  Scope* child = new Scope(this);
  {
    SCOPE_KIDS_WRITER_LOCK
    kids_.push_back(child);
  }
  return *child;
}

Variable* Scope::Var(const std::string& name) {
  SCOPE_VARS_WRITER_LOCK
  return VarInternal(name);
}

Variable* Scope::Var(std::string* name) {
  SCOPE_VARS_WRITER_LOCK
  auto new_name = std::to_string(reinterpret_cast<uintptr_t>(this)) + "." +
                  std::to_string(vars_.size());
  if (name != nullptr) {
    *name = new_name;
  }
  return VarInternal(new_name);
}

Variable* Scope::FindVar(const std::string& name) const {
  SCOPE_VARS_READER_LOCK
  return FindVarInternal(name);
}

Variable* Scope::FindLocalVar(const std::string& name) const {
  SCOPE_VARS_READER_LOCK
  return FindVarLocally(name);
}

const Scope* Scope::FindScope(const Variable* var) const {
  SCOPE_VARS_READER_LOCK
  return FindScopeInternal(var);
}

void Scope::DropKids() {
  SCOPE_KIDS_WRITER_LOCK
  for (Scope* s : kids_) delete s;
  kids_.clear();
}

bool Scope::HasKid(const Scope* scope) const {
  SCOPE_KIDS_READER_LOCK
  auto it = std::find(this->kids_.begin(), this->kids_.end(), scope);
  return it != this->kids_.end();
}

std::vector<std::string> Scope::LocalVarNames() const {
  std::vector<std::string> known_vars;
  {
    SCOPE_VARS_READER_LOCK
    known_vars.reserve(this->vars_.size());
    for (auto& p : vars_) {
      known_vars.emplace_back(p.first);
    }
  }
  return known_vars;
}

void Scope::DeleteScope(Scope* scope) const {
  SCOPE_KIDS_WRITER_LOCK
  auto it = std::find(this->kids_.begin(), this->kids_.end(), scope);
  PADDLE_ENFORCE(it != this->kids_.end(), "%p Cannot find %p as kid scope",
                 this, scope);
  this->kids_.erase(it);
  // When making memory benchmark on Fluid, we have to delete scope sync.
  if (FLAGS_benchmark || FLAGS_eager_delete_scope) {
    delete scope;
  } else {
    Async([scope] { delete scope; });
  }
}

void Scope::EraseVars(const std::vector<std::string>& var_names) {
  std::set<std::string> var_set(var_names.begin(), var_names.end());
  SCOPE_VARS_WRITER_LOCK
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
  SCOPE_VARS_WRITER_LOCK
  RenameInternal(origin_name, new_name);
}

std::string Scope::Rename(const std::string& origin_name) const {
  SCOPE_VARS_WRITER_LOCK
  auto new_name = string::Sprintf("%p.%d", this, vars_.size());
  RenameInternal(origin_name, new_name);
  return new_name;
}

Variable* Scope::VarInternal(const std::string& name) {
  auto* v = FindVarLocally(name);
  if (v != nullptr) return v;
  v = new Variable();
  vars_.emplace(name, std::unique_ptr<Variable>(v));
  VLOG(3) << "Create variable " << name;
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

std::string GenScopeTreeDebugInfo(Scope* root) {
  std::stringstream os;

  if (!root) return "";

  // level traversal
  std::queue<Scope*> queue;
  queue.push(root);

  std::vector<Scope*> scopes;

  while (!queue.empty()) {
    auto* end = queue.back();
    Scope* q = nullptr;
    while (q != end) {
      q = queue.front();
      queue.pop();
      os << q << " ";
      scopes.push_back(q);

      for (auto* c : q->kids()) {
        queue.push(c);
      }
    }
    // end of a level
    os << "\n------------------------------------------\n";
  }

  os << "\nDetails:\n\n";

  for (Scope* q : scopes) {
    os << "====\n";
    os << q << ":\n";
    for (auto& var : q->LocalVarNames()) {
      os << "  - " << var << "\n";
    }
  }

  return os.str();
}

}  // namespace framework
}  // namespace paddle
