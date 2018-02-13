# Design Doc: Parallel_Do in PaddlePaddle

In PaddlePaddle, we use parallel_do primitive to represent multithread data parallel processing.

## Design overview

The definition of a parallel_do op looks like the following

```c++
AddInput(kInputs, "Inputs needed to be split onto different devices").AsDuplicable();
AddInput(kParameters, "Parameters are duplicated over different devices")
    .AsDuplicable();
AddInput(kPlaces, "Devices used for parallel processing");
AddOutput(kOutputs, "Outputs needed to be merged from different devices").AsDuplicable();
AddOutput(kParallelScopes,
          "Container for all local variables in forward pass.");
AddAttr<framework::BlockDesc *>(kParallelBlock,
                                "List of operaters to be executed in parallel");
```

A vanilla implementation of parallel_do can be shown as the following (`|` means single thread and
`||||` means multiple threads)

```
In the forward pass
  |      Split input onto different devices
  |      Copy parameter to onto different devices
  ||||   Compute forward pass in parallel
  |      Merge output from different devices

In the backward pass
  |      Split output@grad onto different devices
  ||||   Compute backward pass in parallel
  |      accumulate param@grad from different devices to the first device
  |      Merge input@grad from different devices
```

This implementation allows to write mixed device program like this

```python
# get embedding feature on CPU
feature = some_cpu_only_op(data)

# parallel processing on multiple GPUs
pd = ParallelDo(gpu_places)
with pd.do():
    read_input(feature)
    prediction = my_net(feature)
    write_output(activation)
prediction = pd()
loss = cross_entropy(prediction, label)
```

## Proformance Imporvement

There are serial places we can make this parallel_do faster.

### forward: split input onto different devices

If the input of the parallel_do is independent from any prior opeartors, we can avoid this step by 
prefetching the input onto different devices in a seperate background thread. And the python code
looks like this.
```python
pd = ParallelDo(gpu_places)
with pd.do():
    feature = pre_fetch(gpu_places)
    prediction = my_net(feature)
    write_output(activation)
```

### forward: Copy parameter to onto different devices

We can avoid this step by making each device have a copy of the parameter. This requires:

1. `fluid.default_start_up_program()` to be run on all devices
1. In the backward, allreduce param@grad at different devices, this requires
    1. `backward.py` add `allreduce` operators at parallel_do_grad
    1. `allreduce` operators need to be called in async mode to achieve maximum throughput
1. apply gradients related op(i.e. cliping, normalization, decay, sgd) on different devices in parallel

By doing so, we also avoided "backward: accumulate param@grad from different devices to the first device"


