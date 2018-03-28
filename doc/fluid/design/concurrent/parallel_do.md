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
          "Scopes for all local variables in forward pass. One scope for each device");
AddAttr<framework::BlockDesc *>(kParallelBlock,
                                "List of operaters to be executed in parallel");
```

A vanilla implementation of parallel_do can be shown as the following (`|` means single thread and
`||||` means multiple threads)

```
In the forward pass
  |      Split input onto different devices
  |      Copy parameter onto different devices
  ||||   Compute forward pass in parallel
  |      Merge output from different devices

In the backward pass
  |      Split output@grad onto different devices
  ||||   Compute backward pass in parallel
  |      accumulate param@grad from different devices to the first device
  |      Merge input@grad from different devices
  |      Copy param@grad to the place of parallel_do_op
```

This implementation allows to write mixed device program like this

```python
W1 = fluid.tensor(size=[100,20], parameter=true)
W2 = fluid.tensor(size=[20,15], parameter=true)

data = layers.data()

gpu_places = layers.get_place(use_gpu=True)
# parallel processing on multiple GPUs
pd = ParallelDo(gpu_places)
with pd.do(input=data):
    prediction = softmax(fc(fc(data, W1), W2))
    write_output(prediction)
prediction = pd()
loss = cross_entropy(prediction, label)
```

And the programDesc are like the following

```
# start_program will be run by executor(CPUPlace), all w1, w2 will be allocated on CPU
start_program
{
  vars: w1, w2
  ops: init(w1), init(w2)
}

main_program
{
block0 {
  vars: data, places, w1, w2, w1_grad, w2_grad,
  ops: data, get_place, parallel_do(block1),
       parallel_do_grad(block2),
       sgd(w2, w2_grad),
       sgd(w1, w1_grad)
}
block1 { # the forward pass
  parent_block: 0
  vars: data, h1, h2, loss
  ops: fc, fc, softmax
}
block2 { # the backward pass
  parent_block: 1
  vars: data_grad, h1_grad, h2_grad, loss_gard, local_w1_grad, local_w2_grad
  ops: softmax_grad,
       fc_grad
       fc_grad
}
}
```

## Performance Imporvement

There are serial places we can make this parallel_do faster.

### forward: split input onto different devices

If the input of the parallel_do is independent from any prior opeartors, we can avoid this step by 
prefetching the input onto different devices in a seperate background thread. And the python code
looks like this.
```python
pd = ParallelDo(gpu_places)
with pd.do():
    feature = get_data_from_prefetch_queue(gpu_places)
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

By doing so, we also avoided "backward: accumulate param@grad from different devices to the first device".
And the ProgramDesc looks like the following

```
# w1, w2 will be allocated on all GPUs
start_program
{
block0 {
  parallel_do(block1)
}
block1 {
  parent_block: 0
  vars: w1, w2
  ops: init(w1), init(w2)
}
}

main_program
{
block0 {
  vars: data, places, w1, w2
  ops: data, get_place, parallel_do(block1),
       parallel_do_grad(block2),      # append_backward
       parallel_do(block3)            # append_optimization
       
}
block1 {
  parent_block: 0
  vars: data, h1, h2, loss
  ops: fc, fc, softmax
}
block2 {
  parent_block: 1
  vars: data_grad, h1_grad, h2_grad, loss_gard, w1_grad, w2_grad
  ops: softmax_grad,
       fc_grad, allreduce(places, scopes, w1_grad),
       fc_grad, allreduce(places, scopes, w2_grad)
}
block3 {
  parent_block: 0
  vars: lr
  ops: sgd(w2, w2_grad),
       sgd(w1, w1_grad)
}
}
```
