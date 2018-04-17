# Design of Scope in Paddle

## Overview

Scope is an important concept in programming languages, which defines a program region that a set of bindings between names and entities applies. In a specific scope, a valid name is uniquely associated with an entity, such as a variable. And in another scope, this name may refer to other entity or nothing at all. It clearly restricts the visibility and validity of names in a program. Hence **Scope** is introduced to PaddlePaddle to manage variables in context. But different from the original abstract concept, Scope now becomes an object with two important attributes:

- Scope is an association of a name to variable.
- Variables in a parent scope can be retrieved from local scope.

A detailed explanation of these two attributes goes as following.


## Scope is an association of a name to variable.

Scope is an association of a name to variable. All variables belong to `Scope`. You need to specify a scope to run a Net, i.e., `net.Run(&scope)`. One net can run in different scopes and update different variable in the scope.


1. Scope only contains a map of a name to variable.

   All parameters, data, states in a Net should be variables and stored inside a scope. Each op should get inputs and outputs to do computation from a scope, such as data buffer, state (momentum) etc.

1. Variable can only be created by Scope and a variable can only be got from Scope. User cannot create or get a variable outside a scope. This is a constraints of our framework, and will keep our framework simple and clear.

1. Scope only contains methods that are used to Create and Get Variables. Scope do not contain Operators and have no information to run them.
    `Net` is designed to drive the computation and Scope only contains a map of variables. There is no computation logic inside a `Scope`. Scope just handles the lifetime management of variables.
    - `Create` is used to create a Variable by its name and add the mapping relation.
    - `Get` is used to find a Variable by name.

1. Every variable only belongs to one certain Scope.

   Variable can not belong to many scopes. If you want to use variables from parent scope, you can use `parent scope`.

1. Scope should destruct all Variables inside it when itself is destructed. User can never store `Variable` pointer somewhere else.

   Because Variable can only be got from Scope. When destroying Scope, we also need to destroy all the Variables in it. If user store `Variable` pointer to private data member or some global variable, the pointer will be an invalid pointer when associated `Scope` is destroyed.

```cpp
class Scope {
 public:
  Variable* Var(const std::string& name);
  const Variable* FindVar(const std::string& name) const;

 private:
    std::unordered_map<std::string, std::unique_ptr<Variable>> vars_;
};
```


## Parent scope and local scope

Just like [scope](https://en.wikipedia.org/wiki/Scope_(computer_science)) in programming languages, `Scope` in the neural network can also be a local scope. There are two attributes about local scope.

1.  We can create local variables in a local scope. When that local scope is destroyed, all local variables should also be destroyed.
2.  Variables in a parent scope can be retrieved from local scopes of that parent scope, i.e., when user get a variable from a scope, it will try to search this variable in current scope. If there is no such variable in the local scope, `scope` will keep searching from its parent, until the variable is found or there is no parent.

```cpp
class Scope {
 public:
  Scope(const std::shared_ptr<Scope>& scope): parent_(scope) {}

  Variable* FindVar(const std::string& name) const {
    auto it = vars_.find(name);
    if (it != vars_.end()) {
      return it->second.get();
    } else if (parent_ != nullptr) {
      return parent_->FindVar(name);
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

## Interface Design

```cpp
class Variable {
 private:
  Variable() = default;
  friend class Scope;
};

class Scope {
 private:
  Scope(const std::shared_ptr<Scope>& parent = nullptr);

 public:
  static std::shared_ptr<Scope> Create(const std::shared_ptr<Scope>& parent = nullptr);

  // return nullptr if not found.
  Variable* FindVar(const std::string& name) const;

  // return if already contains same name variable.
  Variable* Var(const std::string& name);

 private:
  std::shared_ptr<Scope> parent_;
  std::unordered_map<std::string, std::unique_ptr<Variable>> vars_;
};
```
## Only scope can create a variable

To ensure `only scope can create a variable`, we should mark `Variable`'s constructor as a private member function, and Scope is a friend class of Variable. And then only `Var` can construct `Variable`.

## When scope destroyed, all variables inside this scope should be destroyed together

The scope hold unique pointers for all variables. User can `FindVar` from scope, but he should not hold this pointer as a member variable. Because when scope is destroyed, all variables inside this scope will be destroyed together.

## Sharing a parent scope

Local scope contains a `parent_` pointer. It is a linked-list for scopes. Using a `shared_ptr` because when a local scope is using, its parents cannot be destroyed.

Also, as the parent scope is a `shared_ptr`, we can only `Create()` a scope shared pointer. We cannot construct a scope variable, because it cannot be passed to other scope as `parent` pointer.

## Orthogonal interface

`FindVar` will return `nullptr` when `name` is not found. It can be used as `Contains` method. `Var` will return an `Error` when there is a name conflict locally. Combine `FindVar` and `Var`, we can implement `Var` easily.
