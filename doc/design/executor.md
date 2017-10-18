# Executor Design Doc

## Motivation

We use the executor to do the runtime evaluation of a `ProgramDesc`.

## Overview

An executor takes a `ProgramDesc`, a `block_id` and a `Scope`. The `ProgramDesc` is a list of blocks and each block contains the protobuf definition of all the parameters and operators in that block. The `block_id` specifies the entrance block. And the `Scope` is the container of all the variable instances which are persistent throughout different runs.

### What does the executor do?

It evaluates all the operators in the `block_id`th block of a `ProgramDesc`.

### What does the executor NOT do?

It does not do runtime optimization which means it does not intelligently parse the dependency of each op. It also does not choose which op should be run and the order in which all the ops should be run.

It does not do graph partitioning which means it does execute the process of dividing the `ProgramDesc` into several small pieces and executing them on different devices.

## Implementation

`Executor` evaluates a `ProgramDesc`. Essentially, it instantiates Variables and Operators, then runs all the operators in sequence. [[code]](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/executor.cc)
