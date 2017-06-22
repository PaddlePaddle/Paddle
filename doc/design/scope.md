# What is a scope.

## Overview

预期使用场景。

引出Scope的两个属性。
    1. Scope是Variable的Container
    2. Scope可以共享

## Scope is a Container of Variables.

    * Scope contains Variables as it's data member.
    * Scope contains methods that are used to manage Variables, such as Create/Get/Delete.
    * every variable only belong to one certain Scope.
    * Scope should destruct all Variables within it when itself is destructed.
    * Variable can only be created by Scope.
    * Variable can only be got from Scope.

    * Scope do not contains Operators and have no information to run them.

```cpp
class Scope {
 public:
  Variable* CreateVariable(const std::string& name);
  const Variable* GetVariable(const std::string& name) const;
  bool DeleteVariable(const std::string& name);

 private:
    std::unordered_map<std::string, std::shared_ptr<Vairable>> variable_map_;
};
```


## Parent scope and local scope

Just like [scope](https://en.wikipedia.org/wiki/Scope_(computer_science)) in programming languages, `Scope` in the neural network can also be a local scope. There are two attributes about local scope.

1.  We can create local variables in a local scope. When that local scope are destroyed, all local variables should also be destroyed.
2.  Variables in a parent scope can be retrieved from local scopes of that parent scope, i.e., when user get a variable from a scope, it will try to search this variable in current scope. If there is no such variable in the local scope, `scope` will keep searching from its parent, until the variable is found or there is no parent.

```cpp
class Scope {
public:
  Scope(const std::shared_ptr<Scope>& scope): parent_(scope) {}

  Variable* GetVar(const std::string& name) const {
    Variable* var = GetVarLocally(name);
    if (var != nullptr) {
      return var;
    } else if (parent_ != nullptr) {
      return parent_->Get(name);
    } else {
      return nullptr;
    }
  }

private:
  std::shared_ptr<Scope> parent_ {nullptr};
};
```

In `Scope` class, there is a private data member called `parent_`. `parent_` is a smart pointer to its parent scope. When user `Get` a variable by its `name`, the `name` will be searched inside the current scope. If the variable cannot be found locally and parent scope is not a `nullptr`, the variable will be searched inside that parent scope. `parent_` pointer's default value is `nullptr`. It means that the scope is a global scope when `parent_` is nullptr.

A local scope is very useful when we implement Recurrent Neural Network. Each timestep of an RNN should be a `Net`. Each `Net` of timestep (`StepNet` for short) should use an independent local scope. Just like variables in a while loop is inside a local scope in programming languages. By using a single `StepNet` and changing local scope, we can implement an RNN easily.

# Interface Design

```cpp
class Variable {
private:
  Variable() = default;
  friend class Scope;
};

using VariablePtr = std::weak_ptr<Variable>;

class Scope {
private:
  Scope(const std::shared_ptr<Scope>& parent = nullptr);

public:
  static std::shared_ptr<Scope> Create(const std::shared_ptr<Scope>& parent = nullptr);

  // return nullptr if not found.
  VariablePtr GetVariable(const std::string& name) const;

  // return Error if already contains same name variable.
  Error CreateVariable(const std::string& name);

private:
  std::shared_ptr<Scope> parent_;
  std::unordered_map<std::string, std::shared_ptr<Scope>> attrs_;
};
```
## Only scope can create a variable

To ensure `only scope can create a variable`, we should mark `Variable`'s constructor as a private member function, and Scope is a friend class of Variable. And then only `CreateVariable` can construct `Variable`.

## When scope destroyed, all variables inside this scope should be destroyed together

The `VariablePtr` is a `weak_ptr`. `Net` and `Op` can only get a Variable from `Scope`, but cannot hold it. When scope is destroyed, all `VariablePtr`s belong to this Scope will be changed to `nullptr`.

## Sharing a parent scope

Local scope contains a `parent_` pointer. It is a linked-list for scopes. Using a `shared_ptr` because when a local scope is using, its parents cannot be destroyed.

Also, as the parent scope is a `shared_ptr`, we can only `Create()` a scope shared pointer. We cannot construct a scope variable, because it cannot be passed to other scope as `parent` pointer.

## Orthogonal interface

`GetVariable` will return `nullptr` when `name` is not found. It can be used as `Contains` method. `CreateVariable` will return a `Error` when there is a name conflict locally. Combine `GetVariable` and `CreateVariable`, we can implement `CreateOrGetVariable` easily.
