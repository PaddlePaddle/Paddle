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

#pragma once

#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/framework/variable.h"
#include "paddle/platform/macros.h"

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

  /// Create a variable with given name if it doesn't exist.
  Variable* Var(const std::string& name);

  /// Create a variable with a scope-unique name.
  Variable* Var(std::string* name = nullptr);

  /// Find a variable in the scope or any of its ancestors.  Returns
  /// nullptr if cannot find.
  Variable* FindVar(const std::string& name) const;

  const Scope& parent() const { return *parent_; }

  /// Find the scope or an ancestor scope that contains the given variable.
  const Scope* FindScope(const Variable* var) const;

  void DeleteScope(Scope* scope);

  /// Drop all kids scopes belonged to this scope.
  void DropKids();

  // enumerate all the variables current contains.
  std::vector<std::string> GetAllNames(bool recursive = false) const;

  // Rename variable to a new name
  void Rename(const std::string& origin_name,
              const std::string& new_name) const;

  // Rename variable to a new name and return the new name
  std::string Rename(const std::string& origin_name) const;

 private:
  // Call Scope::NewScope for a sub-scope.
  explicit Scope(Scope const* parent) : parent_(parent) {}

  mutable std::unordered_map<std::string, Variable*> vars_;
  mutable std::list<Scope*> kids_;
  Scope const* parent_{nullptr};

  DISABLE_COPY_AND_ASSIGN(Scope);
};
}  // namespace framework
}  // namespace paddle
