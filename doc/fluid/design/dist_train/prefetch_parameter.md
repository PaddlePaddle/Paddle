# Design Doc: Prefetching Parameter From Parameter Server

## Abstract

We propose an approach to pre-fetch the parameters from a Parameter Server while distributed training so that Fluid is able to train a model with a large number of parameters that cannot be stored in one trainer's memory.

## Background

For an embedding layer, the number of trainable parameters may be very large and it is likely that they may not be able to be stored in one trainer's memory. In Fluid distributed training,
the [Distributed Transpiler](./parameter_server.md#distributed-transpiler) would split every parameter into a number of small parameters that are stored on the Parameter Server. Hence, we can pre-fetch the parameters from the specified Parameter Server using the input `Ids`.

## Design

Prior to reading this design, it would be useful for the reader to make themselves familiar with Fluid [Distributed Training Architecture](./distributed_architecture.md) and 
[Parameter Server](./parameter_server.md).

### Partationed Parameter

<img src="src/split_parameter.png" width="400" />

- **Distributed Transpiler** would split the large parameters
(`weight`) into some partitioned parameters (`weight_0`, `weight_1`, `weight_2`) as shown in the
figure above.
- We can use `round-robin` to distribute the partitioned parameter.

### Pre-fetching Parameters

<img src="src/prefetch_parameters.png" width="400" />

- `prefetch_rpc` operator would prefetch the parameter from different Parameter
    Servers using the input `Ids`. We use [SelectedRows](../../../design/selected_rows.md)
    as the received variable type.
- `merge_selected_rows` operator would merge the received parameters into one
    `SelectedRows` variable.

## TODO

- `prefetch_rpc` operator to send rows index and receive SelectedRows variables.
- `lookup_table` need to support `SelectedRows` variable type as input `Weight`.
- Async Update, To avoid slow-node, Async update is important for distributed training,
  we need a design doc and implement it in future.
