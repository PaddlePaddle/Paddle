# ParallelExecutor Design Doc

## Background

We use parallel_do to describe the multi-GPU training. However, this approach would introduce a 
large number of dependencies at initializer, backward, optimizer and memory optimizer. Adding 
device information could solve this problem and but for the time being, we introduce ParallelExecutor 
as a python wrapper of Executor.

## Design

#### Ownership of scope

1. for inserting nccl_com

#### Python level thread

#### optimize with allreduce op

#### No feed & fetch is the list of value from all device
