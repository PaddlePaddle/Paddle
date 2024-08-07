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

#include "glog/logging.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/framework/threadpool.h"

COMMON_DECLARE_bool(benchmark);
COMMON_DECLARE_bool(eager_delete_scope);

#define SCOPE_KIDS_READER_LOCK phi::AutoRDLock auto_lock(&kids_lock_);
#define SCOPE_KIDS_WRITER_LOCK phi::AutoWRLock auto_lock(&kids_lock_);
#define SCOPE_VARS_READER_LOCK phi::AutoRDLock auto_lock(&vars_lock_);
#define SCOPE_VARS_WRITER_LOCK phi::AutoWRLock auto_lock(&vars_lock_);

namespace paddle::framework {
Scope::Scope() : vars_(), kids_() {}
Scope::~Scope() { DropKids(); }  // NOLINT

Scope& Scope::NewScope() const {
  Scope* child = new Scope(this);
  {
    SCOPE_KIDS_WRITER_LOCK
    kids_.push_back(child);
  }
  return *child;
}

std::unique_ptr<Scope> Scope::NewTmpScope() const {
  return std::unique_ptr<Scope>(new Scope(this));
}

Variable* Scope::Var(const std::string& name) {
  // NOTE(xiongkun03): add {} here to unlock. With {}, scope
  // will do callback after unlock.
  Variable* ret = nullptr;
  {
    SCOPE_VARS_WRITER_LOCK
    ret = VarInternal(name);
  }
  return ret;
}

Variable* Scope::Var(std::string* name) {
  Variable* ret = nullptr;
  std::string new_name;
  {
    SCOPE_VARS_WRITER_LOCK
    new_name = std::to_string(reinterpret_cast<uintptr_t>(this)) + "." +
               std::to_string(vars_.size());
    if (name != nullptr) {
      *name = new_name;
    }
    ret = VarInternal(new_name);
  }
  return ret;
}

Variable* Scope::FindVar(const std::string& name) const {
  SCOPE_VARS_READER_LOCK
  return FindVarInternal(name);
}

Variable* Scope::GetVar(const std::string& name) const {
  auto* var = FindVar(name);
  PADDLE_ENFORCE_NOT_NULL(
      var, common::errors::NotFound("Cannot find %s in scope.", name));
  return var;
}

Variable* Scope::FindLocalVar(const std::string& name) const {
  SCOPE_VARS_READER_LOCK
  return FindVarLocally(name);
}

const Scope* Scope::FindScope(const Variable* var) const {
  SCOPE_VARS_READER_LOCK
  return FindScopeInternal(var);
}

const Scope* Scope::FindScope(const std::string& name) const {
  SCOPE_VARS_READER_LOCK
  return FindScopeInternal(name);
}

const Scope* Scope::root() const {
  const Scope* root_scope = this;
  while (root_scope->parent()) {
    root_scope = root_scope->parent();
  }
  return root_scope;
}

void Scope::DropKids() {
  {
    SCOPE_KIDS_WRITER_LOCK
    for (Scope* s : kids_) {
      delete s;
      s = nullptr;
    }
    kids_.clear();
  }
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

std::vector<Variable*> Scope::LocalVars() {
  std::vector<Variable*> known_vars;
  {
    SCOPE_VARS_READER_LOCK
    known_vars.reserve(this->vars_.size());
    for (auto& p : vars_) {
      known_vars.emplace_back(p.second.get());
    }
  }
  return known_vars;
}

void Scope::DeleteScope(Scope* scope) const {
  {
    SCOPE_KIDS_WRITER_LOCK
    auto it = std::find(this->kids_.begin(), this->kids_.end(), scope);
    PADDLE_ENFORCE_NE(it,
                      this->kids_.end(),
                      common::errors::NotFound(
                          "%p is not found in %p as kid scope", scope, this));
    this->kids_.erase(it);
    // When making memory benchmark on Fluid, we have to delete scope sync.
    if (FLAGS_benchmark || FLAGS_eager_delete_scope) {
      delete scope;
    } else {
      phi::Async([scope] { delete scope; });
    }
  }
}

void Scope::EraseVars(const std::vector<std::string>& var_names) {
  {
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
}

void Scope::Rename(const std::string& origin_name,
                   const std::string& new_name) const {
  {
    SCOPE_VARS_WRITER_LOCK
    RenameInternal(origin_name, new_name);
  }
}

std::string Scope::Rename(const std::string& origin_name) const {
  auto new_name = string::Sprintf("%p.%d", this, vars_.size());
  {
    SCOPE_VARS_WRITER_LOCK
    RenameInternal(origin_name, new_name);
  }
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

const Scope* Scope::FindScopeInternal(const std::string& name) const {
  if (vars_.find(name) != vars_.end()) {
    return this;
  }
  return (parent_ == nullptr) ? nullptr : parent_->FindScope(name);
}

void Scope::RenameInternal(const std::string& origin_name,
                           const std::string& new_name) const {
  auto origin_it = vars_.find(origin_name);
  PADDLE_ENFORCE_NE(
      origin_it,
      vars_.end(),
      common::errors::NotFound(
          "Original variable with name %s is not found in the scope.",
          origin_name));
  auto new_it = vars_.find(new_name);
  PADDLE_ENFORCE_EQ(
      new_it,
      vars_.end(),
      common::errors::AlreadyExists(
          "The variable with name %s already exists in the scope.", new_name));
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
  if (it != vars_.end()) {
    return it->second.get();
  }
  return nullptr;
}

void Scope::EraseVarsExcept(const std::unordered_set<Variable*>& vars) {
  SCOPE_VARS_WRITER_LOCK
  for (auto iter = vars_.begin(); iter != vars_.end();) {
    if (vars.count(iter->second.get()) != 0) {
      ++iter;
    } else {
      vars_.erase(iter++);
    }
  }
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

}  // namespace paddle::framework
