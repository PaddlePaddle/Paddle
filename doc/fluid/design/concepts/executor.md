# Executor Design Doc

## Motivation
In [fluid](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/motivation/fluid.md), we encourage the user to use deep learning programming paradigms to describe the training process. When the user-written Python program is executed, it will first create a protobuf message
[`ProgramDesc`](https://github.com/PaddlePaddle/Paddle/blob/a91efdde6910ce92a78e3aa7157412c4c88d9ee8/paddle/framework/framework.proto#L145) that describes the process and is conceptually like an [abstract syntax tree](https://en.wikipedia.org/wiki/Abstract_syntax_tree).

The executor runs the `ProgramDesc` like an interpreter. `ProgramDesc` contains the intrinsics (operators in this case) and variables which will be used, executor explicitly executes the stored precompiled code.

## Overview

An executor takes a `ProgramDesc`, a `block_id` and a `Scope`.  The `ProgramDesc` is a list of blocks and each block contains the protobuf definition of all the parameters and operators in the block. The `block_id` specifies the entrance block. And the `Scope` is the container of all the variable instances, which is persistent throughout different runs.

## Executor

The `Executor` explicitly executes all the intrinsics (operators here) in the `block_id`th block of a `ProgramDesc`. Essentially, it instantiates Variables and Operators, then runs all the operators in sequence one-by-one.
It is very similar to how a push stack frame works when entering a block, following which it cleans up all the temporary variables when a mini-batch is finished. It does not however, have the stack frame pop process.

### The interface
```c++
  Executor(places);
```
A executor does not own any computing resources, a user can only construct an executor using the specified places.

### Running an Executor

```
  void Run(ProgramDesc, Scope, block_id, create_local_scope);
```
An `Executor` only provides a unified way to execute `ProgramDesc`. `ProgramDesc` is the target that will be executed, the `Scope` specifies the variable container, the `block_id` indicates the entrance block and `create_local_scope` is a boolean that states whether it will destroy the temporary variables after the execution is finished.
