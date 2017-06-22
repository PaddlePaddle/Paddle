# What is a scope.

## Overview

Scope is an important concept in programming languages, which defines a program region that a set of bindings between names and entities applies. In a specific scope, a valid name is uniquely associated with an entity, such as a variable. And in another scope, this name may refer to other entity or nothing at all. It clearly restricts the visibility and validity of names in a program. Hence **Scope** is introduced to PaddlePaddle to manage variables in context. But different from the original abstract concept, Scope now becomes an object with two important attributes:

- Scope is a container of variables
- Scope can be inherited or shared

A detailed explanation of these two attributes goes as following.

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

  Variable* Get(const std::string& name) const {
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

# 接口实现

# 各个接口是啥意思，为啥这么设计
