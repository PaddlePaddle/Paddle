#pragma once

#include <unordered_map>
#include <vector>
#include "paddle/framework/variable.h"
#include "paddle/utils/Error.h"

namespace paddle {
namespace framework {

const static Error AlreadyCreated("Variable has already been created");

/**
 * Scope is an association of a name to Variable. All variables belong to
 * `Scope`. You need to specify a scope to run a Net, i.e., `net.Run(&scope)`.
 * One net can run in different scopes and update different variable in the
 * scope.
 */
class Scope {
 public:
  Scope() {}

  explicit Scope(const std::shared_ptr<Scope>& scope) : parent_(scope) {}

  ~Scope() {}

  // Create Variable in this Scope. Return error if Variable already been
  // created.
  Error __must_check CreateVariable(const std::string& name);

  // Get Variable from this Scope, this function will recursive find Variable
  // from it's parent scope. Return nullptr if not found.
  Variable* GetVariable(const std::string& name) const;

  // find and return Variables in the scope it self.
  Variable* GetVarLocally(const std::string& name) const;

  // Get a Variable from Scope, if the Variable is not exist then create it.
  // User should call this function most of time.
  Variable* GetOrCreateVariable(const std::string& name);

  bool HaveVariable(const std::string& name);

 private:
  std::unordered_map<std::string, std::unique_ptr<Variable>> vars_;
  std::shared_ptr<Scope> parent_{nullptr};
};

}  // namespace framework
}  // namespace paddle
