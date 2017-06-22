# What is a scope.

## Overview

预期使用场景。

引出Scope的两个属性。
    1. Scope是Variable的Container
    2. Scope可以共享

## Scope 是一个Variable的Container

解释下为啥Scope是Variable的container。解释下面几个小点的原因。

    * 他只包含variable
    * 每一个variable也只属于一个Scope
    * 每一个Scope析构的时候，会同时析构variable
    * 只能通过Scope创建Vairable。
    * 只能通过Scope获取Variable。

## Parent scope and local scope

Just like [scope](https://en.wikipedia.org/wiki/Scope_(computer_science)) in programming languages, `Scope` in the neural network also can be local. There are two attributes about local scope.

1.  We can create local variables in a local scope, and when that local scope are destroyed, all local variables should also be destroyed.
2.  Variables in a parent scope can be retrieved from that parent scope's local scope, i.e., when user get a variable from a scope, it will search this variable in current scope firstly. If there is no such variable in local scope, `scope` will keep searching from its parent, until the variable is found or there is no parent.

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

In `Scope` class, there is a private data member called `parent_`. `parent_` is a smart pointer to its parent scope. When user `Get` a variable by its `name`, the `name` will be searched locally inside the current scope. If the variable cannot be found locally and parent scope is not a `nullptr`, the variable will be searched inside that parent scope. `parent_` pointer's default value is `nullptr`. It means that the scope is a global scope when `parent_` is nullptr.

A local scope is very useful when we implement Recurrent Neural Network. Each timestep of an RNN should be a `Net`. Each `Net` of timestep (`StepNet` for short) should use an independent local scope. Just like each variable in a while loop is inside a local scope in programming languages. By using a single `StepNet` and changing local scope, we can implement an RNN easily.

# 接口实现

# 各个接口是啥意思，为啥这么设计
