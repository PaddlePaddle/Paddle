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

Just like [scope](https://en.wikipedia.org/wiki/Scope_(computer_science)) in programming languages, `Scope` in the neural network also can be local.  There are two attributes about local scope.

*  We can create local variables in a local scope, and when that local scope are destroyed, all local variables should also be destroyed.
*  Variables in a parent scope can be retrieved from that parent scope's local scope, i.e., when user get a variable from a scope, it will search this variable in current scope firstly. If there is no such variable in local scope, `scope` will keep searching from its parent, until the variable is found or there is no parent.

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

# 接口实现

# 各个接口是啥意思，为啥这么设计
