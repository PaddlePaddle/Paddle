# ParallelExecutor Design Doc

## Introduction

We introduce `ParallelExecutor` to run multi-GPU training in PaddlePaddle Fluid. It supports
1. keeping a copy of the parameters on each GPU
1. allreduce on a separate stream allowing computation and communication overlap

An example of switching single GPU training to multiple GPUs:
```python
cost = your_neural_network()
opt = fluid.optimizer.SGDOptimizer()
opt.minimize(avg_cost)

# change Executor -> ParallelExecutor
exe = fluid.ParallelExecutor(gpu_list=[0, 1])

for iter in xranges(iter_num):
    exe.run()
```

## Design

In the constructor, a list of parameter, whose gradients need to be allreduced, is given.

During the runtime, `ParallelExecutor` starts `#gpu` threads to run each `Executor`. For every
operator run on each GPU, it will automatically sync with different streams when necessary.

```c++
// if op's input is params' grad:
    // sync with allreduce stream
    // e.g. sgd should wait for allreduce to be finished
SyncMultipleStreams(op);

op->Run(*local_scope, place_);

// if op's output is params' grad:
//     sync with computation stream
//     e.g. allreduce shoudl wait for fc_grad to be finished.
SyncMultipleStreams(op);
```


## API

The `ParallelExecutor.run` has similar interface as `Executor.run`. Besides
1. Scope: we don't expose `scope` in `ParallelExecutor.run` since `ParallelExecutor` has its
own scope to maintain NCCL.
1. Feed: we don't expose `feed` in the API either, because the whole point of implementing
parallel_executor is the speed. The input for NN should be implemented in an reader OP.
1. Fetch: we return the fetched value on all GPUs as a list. (e.g. `exe.run(..., fetch=loss)`
with return `[loss_on_gpu0, loss_on_gpu1]`)
