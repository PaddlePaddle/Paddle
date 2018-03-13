# ParallelExecutor Design Doc

## Introduction

We introduce `ParallelExecutor`, as an alternative solution to `ParallelDo`, to run multi-GPU
training in PaddlePaddle Fluid. Essentially, it is a python wrapper of a list of `Executor`s
and `Scopes`s. In addition, it maintains inter-device variables such as NCCL communicator.

To train a neural network multi GPUs, a user only need to make the following modifications in
the code
1. specify `append_all_reduce` flag in the optimizer
1. use `ParallelExecutor` to run the `programDesc`.

```python
cost = your_neural_network()

opt = fluid.optimizer.SGDOptimizer(..., append_all_reduce=True)
opt.minimize(avg_cost)

exe = fluid.ParallelExecutor(gpu_list=[0, 1])
```

## Design

#### ParallelExecutor

A `ParallelExecutor` contains a list of `Executor`s and its associated `Scope`s. All
the `Scope`s are the subscopes of `ParallelExecutor`'s scope, hence they have access
to inter-device variables such as NCCL communicator.

```

                                  /  SubScope 0, contains weights on GPU 0
Scope, contains NCCL Communicator -- SubScope 1, contains weights on GPU 1
                                  \  ...
```

During the runtime, we start `#gpu` python threads to run each `Executor`s.

#### Optimize with AllReduce op

During the construction of the optimization path, AllReduce Op is added to the ProgramDesc.


## API

The `ParallelExecutor.run` has similar interface as `Executor.run`. Besides
1. Scope: we don't expose `scope` in `ParallelExecutor.run` since `ParallelExecutor` has its
own scope to maintain NCCL.
1. Feed: we don't expose `feed` in the API either, because the whole point of implementing
parallel_executor is the speed. The input for NN should be implemented in an reader OP.
1. Fetch: we return the fetched value on all GPUs as a list. (e.g. `exe.run(..., fetch=loss)`
with return `[loss_on_gpu0, loss_on_gpu1]`)
