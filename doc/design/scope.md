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

## Scope 可以被继承或者叫共享

解释下Scope如何被共享，如何查找Variable的算法。
       * Scope永远从本地寻找Variable，找不到会从他的父亲Scope寻找Variable
    * 嵌套深度不做要求。

# 接口实现

C++ code.


## 各个接口是啥意思，为啥这么设计
