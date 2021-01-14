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

#pragma once

extern "C" {
#include <xxhash.h>
}

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/rw_lock.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {

class Scope;

/**
 * @brief Scope that manage all variables.
 *
 * Scope is an association of a name to Variable. All variables belong to
 * Scope. You need to specify a scope to run a Net, i.e., `net.Run(&scope)`.
 * One net can run in different scopes and update different variable in the
 * scope.
 */
class Scope {
 public:
  Scope() {}
  ~Scope();

  /// Create a sub-scope. Returns a reference other than a pointer so
  /// to prevent from manual deletion.
  /// Mark it to const because that new kid scope cannot change parent scope.
  Scope& NewScope() const;

  /// Create a sub-scope for current scope but do not record it in the kids to
  /// avoid performance problems.
  std::unique_ptr<Scope> NewTmpScope() const;

  /// Create a variable with given name if it doesn't exist.
  /// Caller doesn't own the returned Variable.
  Variable* Var(const std::string& name);

  /// Create a variable with a scope-unique name.
  /// Caller doesn't own the returned Variable.
  Variable* Var(std::string* name = nullptr);

  void EraseVars(const std::vector<std::string>& var_names);

  // Erase all variables except the given `vars`
  void EraseVarsExcept(const std::unordered_set<Variable*>& vars);

  /// Find a variable in the scope or any of its ancestors.  Returns
  /// nullptr if cannot find.
  /// Caller doesn't own the returned Variable.
  Variable* FindVar(const std::string& name) const;

  // Get a variable in the scope or any of its ancestors. Enforce
  /// the returned Variable is not nullptr
  Variable* GetVar(const std::string& name) const;

  /// Find a variable in the current scope.
  /// Return nullptr if cannot find.
  /// Caller doesn't own the returned Variable.
  Variable* FindLocalVar(const std::string& name) const;

  const Scope* parent() const { return parent_; }

  /// Find the scope or an ancestor scope that contains the given variable.
  const Scope* FindScope(const Variable* var) const;

  /// Find the scope or an ancestor scope that contains the given variable name.
  const Scope* FindScope(const std::string& name) const;

  void DeleteScope(Scope* scope) const;

  /// Drop all kids scopes belonged to this scope.
  void DropKids();

  /// Find if a scope exists in the kid scopes
  bool HasKid(const Scope* scope) const;

  const std::list<Scope*>& kids() const { return kids_; }

  // enumerate all the variables current contains.
  std::vector<std::string> LocalVarNames() const;

  // Rename variable to a new name
  void Rename(const std::string& origin_name,
              const std::string& new_name) const;

  // Rename variable to a new name and return the new name
  std::string Rename(const std::string& origin_name) const;

 protected:
  struct KeyHasher {
    std::size_t operator()(const std::string& key) const {
      return XXH32(key.c_str(), key.size(), 1);
    }
  };

  mutable std::unordered_map<std::string, std::unique_ptr<Variable>, KeyHasher>
      vars_;

 private:
  // Call Scope::NewScope for a sub-scope.
  explicit Scope(Scope const* parent) : parent_(parent) {}

  // Called by Var.
  Variable* VarInternal(const std::string& name);

  // Called by FindScope.
  const Scope* FindScopeInternal(const Variable* var) const;

  // Called by FindScope.
  const Scope* FindScopeInternal(const std::string& name) const;

  // Called by Rename.
  void RenameInternal(const std::string& origin_name,
                      const std::string& new_name) const;

  // Called by FindVar recursively.
  Variable* FindVarInternal(const std::string& name) const;

  // Called by FindVarInternal and Var.
  Variable* FindVarLocally(const std::string& name) const;

  // Scope in `kids_` are owned by this class.
  mutable std::list<Scope*> kids_;
  const Scope* parent_{nullptr};

  DISABLE_COPY_AND_ASSIGN(Scope);

#ifndef PADDLE_ON_INFERENCE

 private:
  mutable RWLock kids_lock_;
  mutable RWLock vars_lock_;
#endif
};

// Generate some debug string about the inherience structure of scope, quite
// naive.
std::string GenScopeTreeDebugInfo(Scope*);

}  // namespace framework
}  // namespace paddle
