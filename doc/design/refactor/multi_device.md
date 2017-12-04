# Design Doc: How to Execute the ProgramDesc on Multiple Device

A sequence of optimizers will convert a ProgramDesc to ExecutionPlan, which allow
the PaddlePaddle program running on multi devices such CPU, GPU or FPGA,
or multi nodes.

## Running with Multi-CPU

<img src="src/multi_cpu.png">

1. Use Multi-CPU Optimizer to convert the ProgramDesc to ExecutionPlan

    For the data parallelism, we need to pass the attribution `start` and `end`
    index for the `Feed` Op, and this will be calculated in the optimizer step.
1. `Executor` execute the ExecutionPlan which the type is Multi CPU

    the `Executor.run()` will call `block.clone()` and `scope.clone()` to make
    a list of blocks and scopes, the size equals the thread number, and then execute
    the graph in each thread.
1. Collect the gradients and update the parameters

    After all the threads finished their mini-batch, the `Executor` would collector 
    all gradients from the scope list with `sum/mean/...` method, execute the optimizer
    and then copy the result of parameters to each scope.

    Like [nccl](https://developer.nvidia.com/nccl) on GPU, we can implement the same
    the logic to speed up the memory copy.

## Running with Multi-GPU

TODO
