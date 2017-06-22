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

## Scope 可以被继承或者叫共享

解释下Scope如何被共享，如何查找Variable的算法。
       * Scope永远从本地寻找Variable，找不到会从他的父亲Scope寻找Variable
    * 嵌套深度不做要求。

# 接口实现

# 各个接口是啥意思，为啥这么设计
