# Executor Design Doc

## Motivation

We use executor to do the runtime evaluation of a `ProgramDesc`.

## Overview

An executor takes a `ProgramDesc`, a `block_id` and a `Scope`.  The `ProgramDesc` is a list of blocks and each block contains the protobuf definition of all the parameters and operators. The `block_id` specifies the entrance block. And the `Scope` is the container of all the variable instance, which is persistent throughout different runs.

### What does executor do?

It evaluates all the operators in the `block_id`th block of a `ProgramDesc`.

### What does executor NOT do?

It does not do runtime optimization, meaning intelligently parse the dependency of each op a choose which one to be run and in which order they should be run.

It does not do graph partitioning, meaning dividing the `ProgramDesc` into several small pieces and executing them on different devices.

## Implementation

`Executor` evaluates a `ProgramDesc`. Essentially, it instantiates Variables and Operators, then run all the operators in sequence. [[code]](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/executor.cc)
