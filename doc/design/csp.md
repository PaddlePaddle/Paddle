# Design Doc: CSP in PaddlePaddle Fluid

## Motivation

Concurrent programming is important for deep learning.  Few example applications are:

1.  The main thread keeps reading the next mini-batch while another thread uses the GPU for computing.
2.  The main thread performs the computation while another thread uploads the local gradients from each trainer to the parameter server.

Most DL systems, including TensorFlow, Caffe2, and MxNet, can asynchronously execute operators in a graph. However, Fluid doesn't have the concept of a graph at all, as the design goal of Fluid is that of a programming language.

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

It might be a good idea to implement Fluid's select using epoll too.  In this design doc, we start from the O(N) way, so we could focus on Python binding and the syntax.

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

In Go, we first create a channel as explained in the section above and then perform read and write operations on top of the channels.

```go
ch1  := make(chan int)       
ch2  := make(chan int, 100)
```

To write (or perform a `Send` operation) the value of a variable `x`, to channel `ch1` above, we perform the following:

```go
ch1 <- x
fmt.Println("Written to the channel")
```
Now to read (or perform a `Recv` operation) the value stored in `ch2` into a variable `y`, we perform the following:

```go
y <- ch2
fmt.Println("Received from channel")
```

In Fluid, we should be able to perform the above operations on the channel objects as well. As of now, we support two different kinds of channels : [Buffered Channel](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/details/buffered_channel.h) and [UnBuffered Channel](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/framework/details/unbuffered_channel.h)

Send and Receive can be performed as following on a buffered channel:

```python
import threading

def send_to_channel(channel, num_time=1):
  for i in xrange(num_time):
    channel.send(i)

# Create a buffered channel of capacity 10
buffer_size = 10;
ch = fluid.make_channel(dtype=INT, buffer_size)

# Now write three elements to the channel
thread = threading.Thread(target=send_to_channel, args=(ch, 3, ))
thread.daemon = True
thread.start()

# Read all the data from the channel
for i in xrange(3):
  y = ch.recv()

# Done receiving , now close the channel
ch.close()
```

The send and receive operations will be similar for unbuffered channel as well, except for the fact that there is no buffer in an unbuffered channel, so the operations are completely synchronized. For example:

```python
import threading

def send_to_channel(channel, data):
  channel.send(data)

# Create an unbuffered channel
ch = fluid.make_channel(dtype=INT)

# Writes and Reads are synchronous otherwise the calls will block.
thread = threading.Thread(target=send_to_channel, args=(ch, 10, ))
thread.daemon = True
thread.start()

y = ch.recv()

# Done receiving , now close the channel
ch.close()
```

### Select

In Go, the `select` statement lets a goroutine wait on multiple communication operations. A `select` blocks untill one of its cases can run, then it executes that case. It chooses one at random if multiple are ready.

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

- `sel.case(ch1, 'w', X)` : This specifies that we are writing to `ch1` and we want to write the integer in variable `X` to the channel. The character `w` is used here to make the syntax familar to write syntax in Python I/O.

- `sel.case(ch2, 'r', Y)` : This specifies that we would like to read the result from `ch2` into variable `Y`. The character `r` is used here to make the syntax familar to read syntax in Python I/O.

- `sel.default()` : This is equivalent to the default in Go `select`. If none of the channels are ready for read or write, then the fluid code in the default block will be executed.

## Example Programs

### 1. RPC between Trainers and Parameter Servers

### 2. Concurrent Minibatch Loading
