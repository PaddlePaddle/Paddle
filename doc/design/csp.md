# Design Doc: CSP in PaddlePaddle Fluid

## Motivations

Concurrent programming is important for deep learning.  Example applications include

1. A thread uses the GPU for computing while the main thread keeps loading the next minibatch, and
1. a thread uploads the local gradients to the parameter server while the main thread keeps computing.

Most DL systems, including TensorFlow, Caffe2, and MxNet, can asynchronously execute operators in a graph. However, Fluid doesn't have the concept graph at all, as the design goal of Fluid is a programming language.

## Concurrent Programming Models

There were many concurrent programming models, implemented in various forms:

| concurrent programming model | implementation |
|-----|-----|
| mutex | types and functions in standard libraries |
| semaphore | types and functions in standard libraries |
| communicating sequential processes (CSP) | Go programming language |
| actor model | Erlang programming language |
| message passing | MPI |
| bulk synchronous parallel (BSP) | Pregel distributed programming framework |

Because Fluid was designed to be a programming language, we would like to implement CSP.

### CSP v.s. Actor Model

A well-known implementation of Actor Model is the Erlang programming language.  In Actor Model, *processes* could send messages to and receive messages from another process given it ID.  We can find the three ingredients, process with ID, send, and recv, in MPI too.  Indeed, we can rewrite Erlang programs in Python + MPI with possibly fewer lines of code.  Our concern with Actor Model is that it doesn't look reasonable to implement process management in a programming language's runtime library; instead, it seems the OS's responsibility to manage processes and libraries like MPI for send/recv.

## CSP in Fluid

Fluid has two fundamental control-flows: *if-else* and *while*.  If we are to implement CSP, we need:

1. a new data type: *channel*,
1. two new operators: *send* and *recv*, and
1. a new control-flow: select.

Also, we need Python wrappers for the above ingredients.

The type *channel* is conceptually the blocking queue.  In Go, its implemented is a [blocking circular queue](https://github.com/golang/go/blob/68ce117cf17b8debf5754bfd476345779b5b6616/src/runtime/chan.go#L31-L50), which supports send and recv.  The challenge lies more in select.

The operation select has been in OS kernels long before Go language.  All Unix kernels implement system calls *poll* and *select*.  They work by inquiry all file descriptors under their monitoring.  This takes O(N) time.  Since Linux 2.6, a new system call, *epoll*, can do O(1).  In BSD systems, there is a similar system call *kqueue*.  Go's Linux implementation uses epoll.

It might be a great idea to implement Fluid's select using epoll too.  In this design doc, we start from the O(N) way, so could we focus on Python binding and the syntax.

### Type Channel

Fluid supports many data types:

1. Tensor,
1. Row-sparse Tensor
1. LoD Tensor,
1. Tensor array, etc

Each data type is registered in the [`framework.proto`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/framework.proto#L117-L127) as an enum value.  To add a new type channel, we need to add a new type enum.

To expose a C++ type to Python, we need to edit the [`pybind.cc`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/pybind/pybind.cc) file.  [Here](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/pybind/pybind.cc#L120-L164) is an example how we expose C++ class LoDTensor.

## Syntax Design

### Create Channel

In Go, we create a channel by specifying the element type and buffer size:

```go
ch  := make(chan int)       // a channel without buffer
ch1 := make(chan int, 100)  // a channel that can buffer 100 ints.
```

In Fluid, we should be able to do the same:

```python
ch  = fluid.make_chan(dtype=INT)
ch1 = fluid.make_chan(dtype=INT, 100)
```

In addition to that, we want channels that can hold more complex element types, e.g., Tensors of float16:

```python
ch = fluid.make_chan(dtype=Tensor, etype=float16)
```

or Tensors of Tensors of float16 etc.

The point here is that we need a consistent way to compose types, like in C++ we can have `Tensor<Tensor<...<float16>...> >`.

### Send and Recv

### Select

## Exmaple Programs

### 1. RPC between Trainers and Parameter Servers

### 2. Concurrent Minibatch Loading
