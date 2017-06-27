#include "paddle/framework/scope.h"

namespace paddle {
namespace framework {

Error Scope::CreateVariable(const std::string &name) {
  if (name == "") {
    return Error("Variable name should not be empty");
  }

  if (HaveVariable(name)) {
    return AlreadyCreated;
  }
  vars_[name] = std::unique_ptr<Variable>(new Variable());
  return Error();
}

Variable* Scope::GetVarLocally(const std::string& name) const {
  if (vars_.count(name)) {
    return vars_.at(name).get();
  }
  return nullptr;
}

Variable* Scope::GetVariable(const std::string &name) const {
  Variable* var = GetVarLocally(name);
  if (var != nullptr) {
    return var;
  } else if (parent_ != nullptr) {
    return parent_->GetVariable(name);
  } else {
    return nullptr;
  }
}

Variable* Scope::GetOrCreateVariable(const std::string &name) {
  Variable* var;
  var = GetVariable(name);
  if (var == nullptr) {
    auto err = CreateVariable(name);
    if (!err.isOK()) {
      return nullptr;
    }
  }
  return GetVariable(name);
}

bool Scope::HaveVariable(const std::string &name) {
  return vars_.count(name) != 0;
}

}  // namespace framework
}  // namespace paddle

