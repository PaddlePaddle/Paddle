# Design Doc: Parameter Server

## Abstract

We propose an approach to implement the parameter server. In this
approach, there is no fundamental difference between the trainer and
the parameter server: they both run subgraphs, but subgraphs of
different purposes.

## Background

The previous implementations of the parameter server do not run a
fluid sub-program. Parameter initialization, optimizer computation, network
communication and checkpointing are implemented twice on both the
trainer as well as the parameter server.

It would be great if we can write code once and use them on both: the
trainer and the parameter server, since this reduces code duplication and
improves extensibility. Given that after the current refactoring, we are
representing everything as a computation graph on the
trainer. Representing everything as a computation graph on the parameter
server becomes a natural extension.

## Design

### Distributed Transpiler

The *Distributed Transpiler* converts the user-defined fluid program
into sub-programs to be scheduled on different nodes with the following
steps:

1. OP placement: the OPs will be placed on different nodes according
   to a heuristic that minimizes the estimated total computation
   time. Currently we will use a simple heuristic that puts parameter
   variable on parameter server workers and everything else on trainer
   workers.
1. Add communication OPs to enable the communication between nodes.

We will need these OPs: *Send*, *Recv*, *Enqueue*, *Dequeue*.

Below is an example of converting the user defined graph to the
subgraphs for the trainer and the parameter server:

<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/local-graph.png" width="300"/>

After converting:

<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/dist-graph.png" width="700"/>

1. The parameter variable W and its optimizer program are placed on the parameter server.
1. Operators are added to the program.
   - *Send* sends data to the connected *Recv* operator.  The
	 scheduler on the receive node will only schedule *Recv* operator
	 to run when the *Send* operator has ran (the *Send* OP will mark
	 the *Recv* OP runnable automatically).
   - *Enqueue* enqueues the input variable, it can block until space
     become available in the queue.
   - *Dequeue* outputs configurable numbers of tensors from the
     queue. It will block until the queue has the required number of
     tensors.

### Sparse Update

For embedding layers, the gradient may have many rows containing only 0 when training,
if the gradient uses a dense tensor to do parameter optimization,
it could spend unnecessary memory, slow down the calculations and waste
the bandwidth while doing distributed training.
In Fluid, we introduce [SelectedRows](../modules/selected_rows.md) to represent a list of rows containing
non-zero gradient data. So when we do parameter optimization both locally and remotely,
we only need to send those non-zero rows to the optimizer operators:

<img src="https://raw.githubusercontent.com/PaddlePaddle/Paddle/develop/doc/fluid/images/sparse_update.png" width="700" />
### Benefits

- Model parallelism becomes easier to implement: it is an extension to
  the trainer - parameter server approach. We can have several "Transpilers"
  to achieve different goals.
- User-defined optimizer is easier to add - user can now express it as
  a sub-program.
- No more duplication logic inside the trainer and the parameter
  server mentioned in the background section.

### Challenges

- It is important to balance the parameter shards on multiple
  parameter servers. If a single parameter is very big (for example: some
  word-embedding, fully connected, softmax layer), we need to
  automatically partition the single parameter onto different
  parameter servers when possible (only element-wise optimizer depends
  on the parameter variable).
- In the "Async SGD" figure, the "W" variable on the parameter server
  could be read and written concurrently. See
  [here](https://github.com/PaddlePaddle/Paddle/pull/6394) for more
  details about concurrent program in Fluid.

### Discussion

- Can the Enqueue OP be implemented under our current tensor design
  (put the input tensor into the queue tensor)?
- *Dequeue* OP will have variable numbers of output (depending on the
  `min_count` attribute), does our current design support it? (similar
  question for the *Add* OP)

### References

[1] [TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)
