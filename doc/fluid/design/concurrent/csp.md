# Design Doc: CSP in PaddlePaddle Fluid

## Motivation

Concurrent programming is important for deep learning.  Few example applications are:

1.  The main thread keeps reading the next mini-batch while another thread uses the GPU for computing.
2.  The main thread performs the computation while another thread uploads the local gradients from each trainer to the parameter server.

Most DL systems, including TensorFlow, Caffe2, and MxNet, can asynchronously execute operators in a graph. However, Fluid doesn't have the concept of a graph at all, as the design goal of Fluid is that of a programming language.

## Concurrent Programming Models

There were many concurrent programming models, implemented in various forms:

<table>
<thead>
<tr>
<th>concurrent programming model</th>
<th>implementation</th>
</tr>
</thead>
<tbody>
<tr>
<td>mutex </td>
<td>types and functions in standard libraries </td>
</tr>
<tr>
<td>semaphore </td>
<td> types and functions in standard libraries </td>
</tr>
<tr>
<td> communicating sequential processes (CSP)  </td>
<td> Go programming language </td>
</tr>
<tr>
<td> actor model  </td>
<td> Erlang programming language </td>
</tr>
<tr>
<td> message passing  </td>
<td> MPI </td>
</tr>
<tr>
<td> bulk synchronous parallel (BSP)   </td>
<td> Pregel distributed programming framework </td>
</tr>
</tbody>
</table>


Since Fluid was designed to be a programming language, we would like to implement CSP in Fluid.

### CSP v.s. Actor Model

A well-known implementation of Actor Model is the Erlang programming language.  In Actor Model, *processes* could send messages to another process and receive messages from another process given the process IDs.  We can find the three ingredients, process with ID, send, and recv, in MPI too.  Indeed, we can rewrite Erlang programs in Python + MPI with possibly fewer lines of code.  Our concern with Actor Model is that it doesn't seem reasonable to implement process management in a programming language's runtime library; instead, it should be the operating systems' responsibility to manage processes and libraries like MPI for send/recv.

## CSP in Fluid

Fluid has two fundamental control-flows: *if-else* and *while*.  If we are to implement CSP, we need the following:

1. a new data type: *channel* and operators *send* and *recv*,
1. *goroutine* or thread, and
1. a new control-flow: select.

We also need Python wrappers for the above components.

The type *channel* is conceptually the blocking queue.  In Go, its implemented is a [blocking circular queue](https://github.com/golang/go/blob/68ce117cf17b8debf5754bfd476345779b5b6616/src/runtime/chan.go#L31-L50), which supports send and recv.

The `select` operation has been in OS kernels long before Go language.  All Unix kernels implement system calls *poll* and *select*.  They monitor multiple file descriptors to see if I/O is possible on any of them.  This takes O(N) time.  Since Linux 2.6, a new system call, *epoll*, can do the same in O(1) time.  In BSD systems, there is a similar system call *kqueue*.  Go's Linux implementation uses epoll.

It might be a good idea to implement Fluid's select using epoll too.  In this design doc, we start from the O(N) way so that we could focus on Python binding and the syntax.

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
ch  = fluid.make_channel(dtype=INT)
ch1 = fluid.make_channel(dtype=INT, 100)
```

In addition to that, we want channels that can hold more complex element types, e.g., Tensors of float16:

```python
ch = fluid.make_channel(dtype=Tensor, etype=float16)
```

or Tensors of Tensors of float16 etc.

The point here is that we need a consistent way to compose types, like in C++ we can have `Tensor<Tensor<...<float16>...> >`.

### Send and Recv

Go's CSP implementation depends on data type *channel*. There are two types of channels:

1. The unblocked channel, or buffered channel, is a blocking queue with a non-zero sized buffer. The sending to buffered channel blocks if the buffer is full, and the receive operation blocks if the buffer is empty.
1. blocked channel, or unbuffered channel, is a blocking queue with no buffer.  Both sending and receiving block with unbuffered channels.

There are four types of actions with a channel:

1. Create a channel

   ```go
   ch := make(chan int) // this is an unbuffered channel
   ch := make(chan int, 100) // this is a buffered channel of 100 ints.
   ```

1. Send

   ```go
   ch <- 111
   ```

1. Recv

   ```go
   y, ok <- ch
   ```

1. Close

   ```go
   close(ch)
   ```

   Please be aware that a closed channel is not a nil channel, which is `var ch chan int`.

There are some [axioms with channels](https://dave.cheney.net/2014/03/19/channel-axioms):

1. A send to a nil channel blocks forever

1. A receive from a nil channel blocks forever

1. A send to a closed channel panics

1. A receive from a closed channel returns the residual values and then zeros.

In Fluid, we have [buffered channels](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/details/buffered_channel.h) and [unbuffered channels](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/details/unbuffered_channel.h)

The following program illustrates the Python syntax for accessing Fluid buffers.

```python
import fluid

buffer_size = 10
ch = fluid.make_channel(dtype=INT, buffer_size)

# Now write three elements to the channel
with fluid.while(steps=buffer_size):
  fluid.send(ch, step)

fluid.close_channel(ch)

with fluid.while(steps=buffer_size):
  fluid.print(fluid.recv(ch))
```

The following example shows that to avoid the always-blocking behavior of unbuffered channels, we need to use Fluid's goroutines.

```python
import fluid

ch = fluid.make_channel(dtype=INT)

with fluid.go():
  fluid.send(ch)

y = fluid.recv(ch)

fluid.close_channel(ch)
```

### Select

In Go, the `select` statement lets a goroutine wait on multiple communication operations. A `select` blocks until one of its cases can run, then it executes that case. It chooses one at random if multiple are ready.

```go

ch1  := make(chan int)       
ch2  := make(chan int, 100)

x := 0

for {
    select {
    case ch1 <- x:
      x := x + 1
    case y <- ch2:
      fmt.Println("Received on channel")
    default:
      fmt.Println("Default")
    }
  }

```

In Fluid, we should be able to do the same:

```python
ch1  = fluid.make_chan(dtype=INT)
ch2 = fluid.make_chan(dtype=INT, 100)

sel = fluid.select()

with sel.case(ch1, 'w', X):
    fluid.layers.increment(X)

with sel.case(ch2, 'r', Y):
    fluid.print("Received on Channel")

with sel.default():
    fluid.print("Default")

```

In the above code snippet, `X` and `Y` are variables. Now let us look at each of these statements one by one.

- `sel.case(ch1, 'w', X)` : This specifies that we are writing to `ch1` and we want to write the integer in variable `X` to the channel. The character `w` is used here to make the syntax familiar to write syntax in Python I/O.

- `sel.case(ch2, 'r', Y)` : This specifies that we would like to read the result from `ch2` into variable `Y`. The character `r` is used here to make the syntax familiar to read syntax in Python I/O.

- `sel.default()` : This is equivalent to the default in Go `select`. If none of the channels are ready for read or write, then the fluid code in the default block will be executed.

## Example Programs

### 1. RPC between Trainers and Parameter Servers

### 2. Concurrent Minibatch Loading
