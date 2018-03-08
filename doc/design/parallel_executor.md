# ParallelExecutor Design Doc

## Background

We use parallel_do to describe the multi-GPU training. However, this approach would
introduce a large number of dependencies at initializer, backward, optimizer and memory
optimizer. Adding device information could solve this problem and but for the time being,
we introduce ParallelExecutor as a python wrapper of Executor.

## Design

#### API

We don't expose `scope` as in the run interface because we use it to maintaining the inter-device
variables. For example

1. NCCL communicator
1. Data reader(?)

We don't expose feed in parallel_do, because it is time consuming to split feed var
onto different devices, while the whole point of implementing parallel_executor is the speed.

#### Python level thread

#### optimize with allreduce op
