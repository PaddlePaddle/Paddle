# Design Doc: NCCL support in Paddle Fluid

## Abstract

This Design Doc refers to the NCCL feature in  paddle.  We propose an approach to support NCCL library both on a single machine and multiple machines. We wrapper the NCCL primitives `Broadcast`, `Allreduce`, `Reduce` as operators to utilize Multi-GPU powers in one script.


## Motivation

NCCL is a Nvidia library support Multi-GPU communicating. [NCCL](https://developer.nvidia.com/nccl). With NCCL library, we can easily accelerate the training in parallel.

- can easily move the optimize sub-graph to parameter server,  multi-GPU feature can be compatible with distributed support design.
- easily plug-in with [NCCL2](https://developer.nvidia.com/nccl) library.
- GPU Model parallelism becomes easier to implement. we only need to replace different GPU's sub-graph with different part of the whole graph.
- GPU Data Parallelism 

  Suppose to we have `n`GPUs, every GPU has `1/n`part of training data, and store a complete model in GPU memory.  

- GPU Model Parallelism

  every GPU have part of a complete model in GPU memory.

At the beginning of training, the framework needs to issue the same sub-graph to every GPU in Data Parallelism, or different sub-graph in Model Parallelism.

During training, we need the operations of peer to peer copy between different GPUs, aggregating gradients/parameters from GPUs, and broadcasting parameters to GPUs. Every GPU only need to run the sub-graph with correct place information.

Besides, it needs interfaces to synchronize model update with each other, and issue/merge model from different GPU Cards. 

## Implementation

As mentioned above, we summarise that several kinds of operators are needed. Currently, we need to issue parameters to different GPUs,  named it with Broadcast operator.  And also synchronize parameters between GPUs, called it with AllReduce. 

### Graph Converter

To be compatible with [parameter server design doc](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/ops/dist_train.md), the graph converter converts the user defined operation graph into sub-graphs to be executed on different devices.

1. The user-defined operator graph will be partitioned into sub-graph. 

2. Control operators between GPUs will be inserted into the graph.

   *Broadcast, AllReduce in a single machine. And Broadcast, AllReduce, [Send, Recv](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/design/ops/dist_train.md#graph-converter) in multiple machines*

   <img src="images/multigpu_before_convert.png" width="300"/>

After convert, the graph as shows

<img src="images/multigpu_allreduce.png" width="1000"/>

Operators are added to the sub-graphs. Every GPU assigned a role of `rank0`, `rank1` etc. 

- **Broadcast**. Broadcast operator distribute initialized parameter to all the GPUs from the GPU who owns it. e.g. from`rank0` GPU.
- **Allreduce**. Allreduce operator synchronizes parameters/gradients between GPUs. AllReduce implemented in the Ring-Based  communicating method, avoid of the bottle neck in a single GPU.

These two operators need the Multi-GPU context support.

Need to notice that Allreduce operator force GPUs synchronized at that point. Every device only need runs sub-graph in a loop style forever, the whole training process in asynchronous or synchronous mode depends on the Allreduce point in the graph.

As it shown in the picture, when each GPU compute the gradient of `W`, followed with a `AllReduce` operator, accumulate the `dW` to full batch of data, then run the optimize process individually and apply the gradient to its `W`.

In fact, in the way of every GPU optimized full batch of data, wasted (n-1) GPU compute resources. We will enhance it in the next stage.

### Benefits
