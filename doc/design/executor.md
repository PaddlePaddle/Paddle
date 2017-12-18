# Executor Design Doc

## Motivation
In the [fluid](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/fluid.md), we encourage user use deep learning programming paradigms to describe training process. When the user-written Python program is executed, it will create a protobuf message
[`ProgramDesc`](https://github.com/PaddlePaddle/Paddle/blob/a91efdde6910ce92a78e3aa7157412c4c88d9ee8/paddle/framework/framework.proto#L145) that describes the process and is conceptually like an [abstract syntax tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree).

The executor runs the `ProgramDesc` like an interpreter. `ProgramDesc` contains intrinsics/operators and variables which will be used, executor explicitly execute the stored precompiled code.

## Overview

An executor takes a `ProgramDesc`, a `block_id` and a `Scope`.  The `ProgramDesc` is a list of blocks and each block contains the protobuf definition of all the parameters and operators. The `block_id` specifies the entrance block. And the `Scope` is the container of all the variable instance, which is persistent throughout different runs.

## Executor

`Executor` explicitly executes all the intrinsics/operators in the `block_id`th block of a `ProgramDesc`. Essentially, it instantiates Variables and Operators, then runs all the operators in sequence. It is very similar to push stack frame when entering the block, it will destroy the temporary variables when mini-batch is finished, but it does not have stack frame pop process.

### Interface
```c++
  Executor(places);
```
A executor does not own any computing resources, user can only construct an executor with specified places.


```
  void Run(ProgramDesc, Scope, block_id, create_local_scope);
```
A executor only provides an unified way to execute `ProgramDesc`. `ProgramDesc` is the target will be executed, scope specifies the variable container. `block_id` indicates the entrance block, `create_local_scope` means if it will destroy the temporary variables after execution finished.
