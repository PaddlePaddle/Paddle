# Design Doc: Prefecting Parameter From Parameter Server

## Abstract

We propose an approach to prefetch parameter from Parameter
Server while distributed training so that Fluid would training
a model including the large parameter which could not be stored in one
trainer's memory.

## Background

For an embedding layer, the trainable parameter may be very large and could
not be stored in one trainer's memory. In Fluid distributed training,
[Distributed Transpiler](./parameter_server.md#distributed-transpiler) would split every parameter into a number of small
parameters and stored in Parameter Server, so we could prefetch the parameter
from the specified Parameter Server according to the input `Ids`.

## Design

This is a feature of Fluid distributed training, maybe you want
to know [Distributed Architecture](./distributed_architecture.md) and
[Parameter Server](./parameter_server.md) before reading the following content.

### Partationed Parameter

<img src="src/split_parameter.png" width="400" />

- **Distributed Transpiler** would split the large parameter
(weight) into some partitioned parameters (weight_0, weight_1, weight_2) as the
figure above.
- We could use `round-robin` to distribute the partitioned parameter.

### Prefetching Parameter

<img src="src/prefetch_parameters.png" width="400" />

- `prefetch_rpc` operator would prefetch the parameter from different Parameter
    Server according with the input `Ids`, we use [SelectedRows](../../../design/selected_rows.md)
    as the received variable type.
- `merge_selected_rows` operator would merge the received parameters into one
    `SelectedRows` variable.

## TODO

- `prefetch_rpc` operator to send rows index and receive SelectedRows variables.
- `lookup_table` need to support `SelectedRows` variable type as input `Weight`.
- Async Update, To avoid slow-node, Async update is important for distributed training,
  we need a design doc and implement it in future.
