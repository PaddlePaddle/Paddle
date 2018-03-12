# ParallelExecutor Design Doc

## Introduction

We introduce `ParallelExecutor`, as an alternative solution to `ParallelDo`, to run multi-GPU
training in PaddlePaddle Fluid. Essentially, it is a python wrapper of a list of `Executor`s and `Scopes`s. In addition, it
maintains inter-device variables such as NCCL communicator.

The usage of `ParallelExecutor` is the following

To make a neural network be trained on multi GPUs, a user need to specify `append_all_reduce`
flag in the optimizer and use `ParallelExecutor` to run the `programDesc`.

```python
cost = your_neural_network()

opt = fluid.optimizer.SGDOptimizer(..., append_all_reduce=True)
opt.minimize(avg_cost)

exe = fluid.ParallelExecutor(gpu_list=[0, 1])
```

High performance multi-GPU training requires parameters being replicated on each
device. We introduce `ParallelExecutor` as a python wrapper of `Executor`.

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
