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

#include <list>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/temp_variable_pool.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {

int64_t GetEagerDeletionThreshold();
bool IsFastEagerDeletionModeEnabled();

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
  Scope() : Scope(nullptr) {}

  ~Scope();

  /// Store temporary variables. These variables would be deleted when scope
  /// is deleted.
  void AddTempVar(std::unique_ptr<Variable>&& var) const;

  /// Create a sub-scope. Returns a reference other than a pointer so
  /// to prevent from manual deletion.
  /// Mark it to const because that new kid scope cannot change parent scope.
  Scope& NewScope() const;

  /// Create a variable with given name if it doesn't exist.
  /// Caller doesn't own the returned Variable.
  Variable* Var(const std::string& name);

  /// Create a variable with a scope-unique name.
  /// Caller doesn't own the returned Variable.
  Variable* Var(std::string* name = nullptr);

  void EraseVars(const std::vector<std::string>& var_names);

  /// Find a variable in the scope or any of its ancestors.  Returns
  /// nullptr if cannot find.
  /// Caller doesn't own the returned Variable.
  Variable* FindVar(const std::string& name) const;

  /// Find a variable in the current scope.
  /// Return nullptr if cannot find.
  /// Caller doesn't own the returned Variable.
  Variable* FindLocalVar(const std::string& name) const;

  const Scope* parent() const { return parent_; }

  /// Find the scope or an ancestor scope that contains the given variable.
  const Scope* FindScope(const Variable* var) const;

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
  mutable std::unordered_map<std::string, std::unique_ptr<Variable>> vars_;

 private:
  // Call Scope::NewScope for a sub-scope.
  explicit Scope(Scope const* parent) : parent_(parent) {
    InitTempVariablePool();
  }

  // Init TempVariablePool. This function can only be called
  // inside constructor, because it is not thread-safe
  void InitTempVariablePool();

  // Called by Var.
  Variable* VarInternal(const std::string& name);

  // Called by FindScope.
  const Scope* FindScopeInternal(const Variable* var) const;

  // Called by Rename.
  void RenameInternal(const std::string& origin_name,
                      const std::string& new_name) const;

  // Called by FindVar recursively.
  Variable* FindVarInternal(const std::string& name) const;

  // Called by FindVarInternal and Var.
  Variable* FindVarLocally(const std::string& name) const;

  // Scope in `kids_` are owned by this class.
  mutable std::list<Scope*> kids_;
  const Scope* parent_;

  // Temp variable pool
  mutable std::weak_ptr<TempVariablePool> tmp_vars_;

  DISABLE_COPY_AND_ASSIGN(Scope);

 private:
  mutable std::mutex mutex_;

  template <typename LockType>
  struct LockGuard {
#ifdef PADDLE_ON_INFERENCE
    // When in inference scenario, the scopes will not be written by two threads
    // in a mean time, but a scope may be read by multiple threads concurrently,
    // and the mutex will cause serious performance issue.
    // So the mutex is disabled when defined `PADDLE_ON_INFERENCE`.
    inline explicit LockGuard(LockType& mtx) {}  // NOLINT
#else
    inline explicit LockGuard(LockType& mtx) : guard_(mtx) {}  // NOLINT
    std::lock_guard<LockType> guard_;
#endif
  };
};

// Generate some debug string about the inherience structure of scope, quite
// naive.
std::string GenScopeTreeDebugInfo(Scope*);

}  // namespace framework
}  // namespace paddle
